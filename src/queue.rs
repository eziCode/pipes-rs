use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

#[derive(Clone, Copy, Debug)]
pub enum QueuePolicy {
    Backpressure,
    DropOldest,
}

pub enum PushResult<T> {
    Accepted { depth: usize, dropped: Option<T> },
    Closed,
}

struct State<T> {
    items: VecDeque<T>,
    closed: bool,
}

pub struct BoundedQueue<T> {
    capacity: usize,
    policy: QueuePolicy,
    state: Mutex<State<T>>,
    not_empty: Condvar,
    not_full: Condvar,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize, policy: QueuePolicy) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            policy,
            state: Mutex::new(State {
                items: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    pub fn push(&self, item: T) -> PushResult<T> {
        let mut state = self.state.lock().expect("queue mutex poisoned");
        if matches!(self.policy, QueuePolicy::Backpressure) {
            while state.items.len() == self.capacity && !state.closed {
                state = self.not_full.wait(state).expect("queue mutex poisoned");
            }
        }
        if state.closed {
            return PushResult::Closed;
        }
        let dropped = if state.items.len() == self.capacity {
            state.items.pop_front()
        } else {
            None
        };
        state.items.push_back(item);
        let depth = state.items.len();
        self.not_empty.notify_one();
        PushResult::Accepted { depth, dropped }
    }

    pub fn pop(&self) -> Option<T> {
        let mut state = self.state.lock().expect("queue mutex poisoned");
        while state.items.is_empty() && !state.closed {
            state = self.not_empty.wait(state).expect("queue mutex poisoned");
        }
        let item = state.items.pop_front();
        if item.is_some() {
            self.not_full.notify_one();
        }
        item
    }

    pub fn close(&self) {
        let mut state = self.state.lock().expect("queue mutex poisoned");
        state.closed = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_oldest_keeps_latest_items() {
        let queue = BoundedQueue::new(2, QueuePolicy::DropOldest);
        assert!(matches!(
            queue.push(1),
            PushResult::Accepted { dropped: None, .. }
        ));
        assert!(matches!(
            queue.push(2),
            PushResult::Accepted { dropped: None, .. }
        ));
        assert!(matches!(
            queue.push(3),
            PushResult::Accepted {
                dropped: Some(1),
                ..
            }
        ));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
    }
}
