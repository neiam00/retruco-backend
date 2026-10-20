.PHONY: dev test check fmt lint

dev:
	APP_ENV=development go run ./cmd/server

test:
	go test -race ./...

check:
	go vet ./...
	go test ./...

fmt:
	gofmt -w ./rules

lint:
	golangci-lint run ./...
