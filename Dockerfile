# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Multi-architecture index digests freeze both image content and the mapping
# from TARGETPLATFORM to its platform-specific image.
ARG RUST_IMAGE=rust:1.94.0-trixie@sha256:f17e723020f87c1b4ac4ff6d73c9dfbb7d5cb978754c76641e47337d65f61e12
ARG RUNTIME_IMAGE=debian:trixie-slim@sha256:d7e12182ce18b85b93007c1dedf31f2d29e01ccf3182cc4017c709b6259bc132

FROM ${RUST_IMAGE} AS builder
WORKDIR /build

COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    mkdir src \
    && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --locked --release \
    && rm -rf src

COPY src ./src
COPY assets ./assets
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    touch src/main.rs \
    && cargo build --locked --release \
    && cp target/release/pipes-rs /tmp/pipes-rs \
    && strip /tmp/pipes-rs

FROM ${RUNTIME_IMAGE}
LABEL org.opencontainers.image.title="pipes-rs benchmark"
LABEL org.opencontainers.image.description="Reproducible Waymo Arrow pipeline benchmark"

COPY --from=builder /tmp/pipes-rs /usr/local/bin/pipes-rs
COPY docker/benchmark-entrypoint.sh /usr/local/bin/benchmark-entrypoint

ENTRYPOINT ["/usr/local/bin/benchmark-entrypoint"]
