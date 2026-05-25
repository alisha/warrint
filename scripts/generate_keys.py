#!/usr/bin/env python3
"""Generate an Ed25519 key pair for use with court.py and provider.py."""
import argparse

from Cryptodome.PublicKey import ECC


def main(private_key_path, public_key_path):
  key = ECC.generate(curve='Ed25519')
  with open(private_key_path, 'wb') as f:
    f.write(key.export_key(format='PEM').encode())
  with open(public_key_path, 'wb') as f:
    f.write(key.public_key().export_key(format='PEM').encode())
  print(f"Private key saved to {private_key_path}")
  print(f"Public key saved to {public_key_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate Ed25519 key pair")
  parser.add_argument("--private-key-path", default="examples/example_private_key.pem",
                      help="Output path for private key (default: examples/example_private_key.pem)")
  parser.add_argument("--public-key-path", default="examples/example_public_key.pem",
                      help="Output path for public key (default: examples/example_public_key.pem)")
  args = parser.parse_args()
  main(args.private_key_path, args.public_key_path)
