FROM golang:1.27.4-alpine AS toolchain
RUN apk add --no-cache build-base
WORKDIR /src

# The default image builds the server and migration tool from source.
FROM toolchain AS build
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY cpu ./cpu
COPY internal ./internal
COPY rules ./rules
RUN CGO_ENABLED=0 go build -buildvcs=false -trimpath -o /out/ ./cmd/server ./cmd/migrate

FROM alpine:3.23 AS runtime-base
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 retruco
WORKDIR /app
COPY migrations ./migrations
USER retruco
EXPOSE 8080
CMD ["./server"]

# Alternatively, supply tested executables in a named "binaries" build context.
FROM runtime-base AS ci-runtime
COPY --from=binaries /server /migrate ./

# Keep the default Docker/Compose build self-contained for local installations.
FROM runtime-base AS runtime
COPY --from=build /out/server /out/migrate ./
