//go:build js && wasm

// The browser loads this program once in a dedicated Worker. Calls cross the
// JavaScript/Go boundary once per decision, never once per card comparison.
package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/neiam00/retruco-backend/cpu"
)

func main() {
	decide := js.FuncOf(func(_ js.Value, args []js.Value) any {
		response := cpu.Response{Version: cpu.ProtocolVersion}
		if len(args) != 1 || args[0].Type() != js.TypeString || len(args[0].String()) > 256*1024 {
			response.Error = "Solicitud CPU inválida."
		} else {
			var request cpu.Request
			if json.Unmarshal([]byte(args[0].String()), &request) != nil {
				response.Error = "No pudimos leer la vista CPU."
			} else {
				response = cpu.Decide(request)
			}
		}
		encoded, err := json.Marshal(response)
		if err != nil {
			return `{"version":1,"error":"No pudimos preparar la jugada CPU."}`
		}
		return string(encoded)
	})
	defer decide.Release()
	js.Global().Set("retrucoCPUDecide", decide)
	select {}
}
