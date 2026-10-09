#include "bridge.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Immutable point/box buffers remain in the Rust-owned dataset. Both transports
// transfer a frame reference; neither serializes or copies point payloads.
struct FrameRef { uint64_t index, started; const Dataset* context; };
static_assert(sizeof(FrameRef)==24,"unexpected frame-reference layout");
template<> struct rclcpp::TypeAdapter<FrameRef,std_msgs::msg::UInt64> {
  using is_specialized=std::true_type;
  using custom_type=FrameRef;
  using ros_message_type=std_msgs::msg::UInt64;
  static void convert_to_ros_message(const custom_type&,ros_message_type&) {
    throw std::runtime_error("Unexpected ROS serialization: this pilot requires intra-process delivery");
  }
  static void convert_to_custom(const ros_message_type&,custom_type&) {
    throw std::runtime_error("Unexpected ROS deserialization");
  }
};
using Adapted=rclcpp::TypeAdapter<FrameRef,std_msgs::msg::UInt64>;
static void require(bool good,const char* message) { if(!good) throw std::runtime_error(message); }

static void composed(const Dataset* data,size_t frames,size_t capacity,std::vector<Sample>& rows) {
  auto options=rclcpp::NodeOptions().use_intra_process_comms(true)
    .enable_rosout(false).start_parameter_services(false).start_parameter_event_publisher(false);
  auto source=std::make_shared<rclcpp::Node>("prepared_waymo_source",options);
  auto fusion=std::make_shared<rclcpp::Node>("depth_association",options);
  auto qos=rclcpp::QoS(rclcpp::KeepLast(capacity)).reliable().durability_volatile();
  auto publisher=source->create_publisher<Adapted>("prepared_frame",qos);
  std::mutex mutex; std::condition_variable credit;
  size_t completed=0; bool failed=false;
  std::exception_ptr failure;
  std::vector<const FrameRef*> addresses(frames);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto subscriber=fusion->create_subscription<Adapted>("prepared_frame",qos,
    [&](std::unique_ptr<FrameRef> message) {
      const auto received=px_now();
      try {
        require(message->context==data && message->index<frames,"invalid input reference");
        const auto i=message->index;
        require(message.get()==addresses[i],"intra-process envelope was copied");
        require(i==completed,"missing, duplicated or reordered frame");
        auto result=px_fuse(data,i);const auto finished=px_now();
        require(px_validate_free(data,i,result)==0,"shared-kernel output mismatch");
        rows[i]={i,message->started,received,finished};
        {std::lock_guard<std::mutex> lock(mutex);++completed;}
        credit.notify_one();
      } catch(...) {
        {std::lock_guard<std::mutex> lock(mutex);failed=true;failure=std::current_exception();}
        credit.notify_one();executor.cancel();
      }
    });
  executor.add_node(source);executor.add_node(fusion);
  require(publisher->get_intra_process_subscription_count()==1,"intra-process subscriber not registered");
  std::thread worker([&] {
    try {executor.spin();}
    catch(...) {
      {std::lock_guard<std::mutex> lock(mutex);failed=true;failure=std::current_exception();}
      credit.notify_one();
    }
  });
  try {
    for(size_t i=0;i<frames;++i) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        require(credit.wait_for(lock,std::chrono::seconds(60),[&]{return failed||i<completed+capacity;}),"credit timeout");
        if(failed) std::rethrow_exception(failure);
      }
      const auto started=px_now();
      auto msg=std::make_unique<FrameRef>(FrameRef{i,started,data});
      addresses[i]=msg.get();publisher->publish(std::move(msg));
    }
    std::unique_lock<std::mutex> lock(mutex);
    require(credit.wait_for(lock,std::chrono::seconds(60),[&]{return failed||completed==frames;}),"completion timeout");
    if(failed) std::rethrow_exception(failure);
  } catch(...) {executor.cancel();worker.join();throw;}
  executor.cancel();worker.join();
}
int main(int argc,char** argv) {
  // Positional arguments deliberately avoid mixing benchmark flags with ROS flags.
  if(argc!=9) {std::cerr<<"usage: execution-pilot native|ros2 DATA SPLIT SEGMENT FRAMES CAPACITY OUTPUT.csv RUN.json\n";return 2;}
  try {
    const std::string mode=argv[1];require(mode=="native"||mode=="ros2","invalid mode");
    const size_t frames=std::stoull(argv[5]),capacity=std::stoull(argv[6]);
    require(frames>0&&capacity>0,"positive frames and capacity required");
    std::unique_ptr<Dataset,decltype(&px_free)> data(px_load(argv[2],argv[3],argv[4],frames),px_free);
    require(data!=nullptr,"dataset preparation failed");
    std::vector<Sample> rows(frames);
    // ROS context init is outside graph_call_ns; node/thread setup is inside.
    // Per-frame active intervals exclude setup. Both use the same executable.
    if(mode=="ros2") rclcpp::init(0,nullptr);
    const auto start=px_now();
    if(mode=="native") require(px_native(data.get(),capacity,rows.data(),rows.size())==0,"native execution failed");
    else composed(data.get(),frames,capacity,rows);
    const auto finish=px_now();
    if(mode=="ros2") rclcpp::shutdown();
    require(px_write(data.get(),rows.data(),rows.size(),argv[7])==0,"output checks failed");
    std::ofstream meta(argv[8]);meta<<"{\"mode\":\""<<mode<<"\",\"frames\":"<<frames
      <<",\"capacity\":"<<capacity<<",\"graph_call_ns\":"<<finish-start
      <<",\"validation\":\"exact depth bits and counts\",\"representation\":\"native Vec<Point> behind immutable frame references\"}\n";
    require(meta.good(),"metadata write failed");
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
