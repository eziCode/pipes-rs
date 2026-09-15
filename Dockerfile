# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Multi-architecture index digests freeze both image content and the mapping
# from TARGETPLATFORM to its platform-specific image.
ARG RUST_IMAGE=rust:1.94.0-bookworm@sha256:365468470075493dc4583f47387001854321c5a8583ea9604b297e67f01c5a4f
ARG RUNTIME_IMAGE=debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171

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
