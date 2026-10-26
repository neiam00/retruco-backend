package api

import (
	"crypto/subtle"
	"encoding/json"
	"io"
	"net"
	"net/http"
)

// This operator endpoint is reachable only over the container's loopback
// interface with the configured proxy secret. Forwarded client headers are
// deliberately ignored; public browsers and the edge cannot announce updates.
func (s *Server) deployment(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	ip := net.ParseIP(host)
	if err != nil || ip == nil || !ip.IsLoopback() || s.config.ProxySecret == "" ||
		subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Retruco-Proxy-Secret")), []byte(s.config.ProxySecret)) != 1 {
		fail(w, http.StatusForbidden, "Solicitud no permitida.")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		fail(w, http.StatusMethodNotAllowed, "Método no permitido.")
		return
	}
	var request struct {
		Updating *bool `json:"updating"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 256))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&request) != nil || request.Updating == nil || decoder.Decode(new(any)) != io.EOF {
		fail(w, http.StatusBadRequest, "Aviso de actualización inválido.")
		return
	}
	s.host.SetUpdating(*request.Updating)
	writeJSON(w, http.StatusOK, map[string]bool{"updating": s.host.Updating()})
}
