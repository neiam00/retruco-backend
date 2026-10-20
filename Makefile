.PHONY: dev test check fmt lint browser-engine test-wasm

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

# Publish cpu.wasm together with the runtime from the exact Go toolchain used.
browser-engine:
	mkdir -p bin/browser-engine
	GOOS=js GOARCH=wasm go build -trimpath -buildvcs=false -ldflags="-s -w" -o bin/browser-engine/cpu.wasm ./cmd/cpu
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" bin/browser-engine/wasm_exec.js
	@if test -f "$$(go env GOROOT)/LICENSE"; then \
		cp "$$(go env GOROOT)/LICENSE" bin/browser-engine/LICENSE.GO; \
	else \
		cp "$$(go env GOROOT)/../LICENSE" bin/browser-engine/LICENSE.GO; \
	fi

test-wasm:
	GOOS=js GOARCH=wasm go test -exec="$$(go env GOROOT)/lib/wasm/go_js_wasm_exec" ./rules ./cpu
