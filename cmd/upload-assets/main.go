// upload-assets publishes the generated WebP deck to an existing B2 bucket.
// It is an explicit deployment command; the game server never uploads assets.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/Backblaze/blazer/b2"
)

func main() {
	if err := run(); err != nil {
		slog.Error("asset upload failed", "error", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 2 {
		return errors.New("usage: upload-assets /path/to/frontend/static/cards")
	}
	keyID := os.Getenv("B2_APPLICATION_KEY_ID")
	key := os.Getenv("B2_APPLICATION_KEY")
	bucketName := os.Getenv("B2_BUCKET_NAME")
	if keyID == "" || key == "" || bucketName == "" {
		return errors.New("B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY and B2_BUCKET_NAME are required")
	}

	files, err := filepath.Glob(filepath.Join(os.Args[1], "*.webp"))
	if err != nil {
		return err
	}
	if len(files) != 41 {
		return fmt.Errorf("expected the 40-card deck and one back; found %d WebP images", len(files))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	client, err := b2.NewClient(ctx, keyID, key)
	if err != nil {
		return fmt.Errorf("authorize B2: %w", err)
	}
	bucket, err := client.Bucket(ctx, bucketName)
	if err != nil {
		return fmt.Errorf("open configured bucket: %w", err)
	}
	for _, path := range files {
		if err := upload(ctx, bucket, path); err != nil {
			return err
		}
		slog.Info("uploaded card", "name", filepath.Base(path))
	}
	return nil
}

func upload(ctx context.Context, bucket *b2.Bucket, path string) error {
	// #nosec G304 G703 -- This operator-run CLI reads WebP images from the directory explicitly selected by the operator.
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	uploadContext, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	object := bucket.Object("cards/" + filepath.Base(path))
	writer := object.NewWriter(uploadContext, b2.WithAttrsOption(&b2.Attrs{
		ContentType: "image/webp",
		Info:        map[string]string{"b2-cache-control": "public, max-age=86400"},
	}))
	if _, err := io.Copy(writer, file); err != nil {
		cancel()
		_ = writer.Close()
		return fmt.Errorf("upload %s: %w", filepath.Base(path), err)
	}
	if err := writer.Close(); err != nil {
		return fmt.Errorf("finish %s: %w", filepath.Base(path), err)
	}
	return nil
}
