.PHONY: test check fmt lint

test:
	go test -race ./...

check:
	go vet ./...
	go test ./...

fmt:
	gofmt -w ./rules

lint:
	golangci-lint run ./...
