FROM golang:1.27.2-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server && \
    CGO_ENABLED=0 go build -trimpath -o /out/migrate ./cmd/migrate

FROM alpine:3.23
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 retruco
WORKDIR /app
COPY --from=build /out/server /out/migrate ./
COPY migrations ./migrations
USER retruco
EXPOSE 8080
CMD ["./server"]
