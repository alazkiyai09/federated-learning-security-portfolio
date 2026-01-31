"""ECDSA signature manager for model updates."""

import base64
import hashlib
import json
from typing import Dict, Any, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from signguard.core.types import ModelUpdate, SignedUpdate


class SignatureManager:
    """Manages ECDSA signatures for model updates.

    Provides signing and verification capabilities using elliptic curve
    digital signatures (ECDSA) for authenticating federated learning updates.
    """

    def __init__(self, curve: ec.EllipticCurve = ec.SECP256R1()):
        """Initialize signature manager.

        Args:
            curve: Elliptic curve to use for signatures (default: SECP256R1/P-256)
        """
        self.curve = curve
        self.backend = default_backend()

    def generate_keypair(self) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        """Generate new ECDSA key pair.

        Returns:
            Tuple of (private_key, public_key)
        """
        private_key = ec.generate_private_key(self.curve, self.backend)
        public_key = private_key.public_key()
        return private_key, public_key

    def sign_update(
        self,
        update: ModelUpdate,
        private_key: ec.EllipticCurvePrivateKey,
    ) -> str:
        """Create ECDSA signature over serialized model update.

        The update is serialized to a canonical JSON format, hashed with SHA-256,
        and then signed with the client's private key.

        Args:
            update: Model update to sign
            private_key: Client's private key

        Returns:
            Base64-encoded signature (DER format)
        """
        # Serialize update to canonical format
        message = self._serialize_update(update)

        # Hash the message
        digest = hashlib.sha256(message.encode()).digest()

        # Sign the digest
        signature = private_key.sign(digest, ec.ECDSA(hashes.SHA256()))

        # Return as base64 string
        return base64.b64encode(signature).decode('utf-8')

    def verify_update(
        self,
        signed_update: SignedUpdate,
    ) -> bool:
        """Verify ECDSA signature on a signed update.

        Args:
            signed_update: Signed update containing signature and public key

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Deserialize public key
            public_key = self.deserialize_public_key(signed_update.public_key)

            # Serialize the update to canonical format
            message = self._serialize_update(signed_update.update)

            # Hash the message
            digest = hashlib.sha256(message.encode()).digest()

            # Decode signature
            signature = base64.b64decode(signed_update.signature)

            # Verify signature
            public_key.verify(signature, digest, ec.ECDSA(hashes.SHA256()))
            return True

        except (InvalidSignature, ValueError, Exception):
            return False

    def serialize_public_key(self, public_key: ec.EllipticCurvePublicKey) -> str:
        """Convert public key to base64 string.

        Args:
            public_key: Public key object

        Returns:
            Base64-encoded public key in PEM format
        """
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return base64.b64encode(pem).decode('utf-8')

    def deserialize_public_key(self, key_str: str) -> ec.EllipticCurvePublicKey:
        """Convert base64 string to public key object.

        Args:
            key_str: Base64-encoded PEM public key

        Returns:
            Public key object

        Raises:
            ValueError: If key string is invalid
        """
        try:
            pem_bytes = base64.b64decode(key_str)
            public_key = serialization.load_pem_public_key(pem_bytes, backend=self.backend)
            
            # Verify it's an EC key
            if not isinstance(public_key, ec.EllipticCurvePublicKey):
                raise ValueError("Key is not an elliptic curve public key")
                
            return public_key
            
        except Exception as e:
            raise ValueError(f"Failed to deserialize public key: {e}")

    def serialize_private_key(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        password: bytes | None = None,
    ) -> str:
        """Convert private key to base64 string.

        Args:
            private_key: Private key object
            password: Optional password for encryption

        Returns:
            Base64-encoded private key in PEM format
        """
        encryption = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
        return base64.b64encode(pem).decode('utf-8')

    def deserialize_private_key(
        self,
        key_str: str,
        password: bytes | None = None,
    ) -> ec.EllipticCurvePrivateKey:
        """Convert base64 string to private key object.

        Args:
            key_str: Base64-encoded PEM private key
            password: Optional password for decryption

        Returns:
            Private key object

        Raises:
            ValueError: If key string is invalid
        """
        try:
            pem_bytes = base64.b64decode(key_str)
            private_key = serialization.load_pem_private_key(
                pem_bytes,
                password=password,
                backend=self.backend,
            )
            
            # Verify it's an EC key
            if not isinstance(private_key, ec.EllipticCurvePrivateKey):
                raise ValueError("Key is not an elliptic curve private key")
                
            return private_key
            
        except Exception as e:
            raise ValueError(f"Failed to deserialize private key: {e}")

    def _serialize_update(self, update: ModelUpdate) -> str:
        """Serialize model update to canonical JSON string.

        Creates a deterministic serialization that includes all relevant
        update data for signing.

        Args:
            update: Model update to serialize

        Returns:
            Canonical JSON string
        """
        # Create canonical representation
        data = {
            "client_id": update.client_id,
            "round_num": update.round_num,
            "parameters": {
                name: self._tensor_to_list(param)
                for name, param in sorted(update.parameters.items())
            },
            "num_samples": update.num_samples,
            "metrics": dict(sorted(update.metrics.items())),
            "timestamp": update.timestamp,
        }
        
        # Serialize with sorted keys for determinism
        return json.dumps(data, sort_keys=True)

    def _tensor_to_list(self, tensor) -> list:
        """Convert tensor to list for serialization.

        Args:
            tensor: PyTorch tensor

        Returns:
            List representation
        """
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().tolist()
        return list(tensor)

    def get_key_info(self, public_key: ec.EllipticCurvePublicKey) -> Dict[str, Any]:
        """Get information about a public key.

        Args:
            public_key: Public key object

        Returns:
            Dictionary with key information
        """
        # Get curve name
        curve_name = public_key.curve.name if hasattr(public_key, 'curve') else 'unknown'
        
        # Get key size in bits
        key_size = public_key.key_size if hasattr(public_key, 'key_size') else 0
        
        # Get key coordinates (for debugging/validation)
        public_numbers = public_key.public_numbers()
        
        return {
            "curve": curve_name,
            "key_size_bits": key_size,
            "x": hex(public_numbers.x),
            "y": hex(public_numbers.y),
        }
