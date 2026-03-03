# security.py - Create this new file
from cryptography.fernet import Fernet
import os

# Generate encryption key (first time only)
def generate_key():
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    return key

# Load encryption key
def load_key():
    if not os.path.exists("secret.key"):
        return generate_key()
    return open("secret.key", "rb").read()

# Encrypt text before saving to database
def encrypt_text(text):
    key = load_key()
    f = Fernet(key)
    encrypted = f.encrypt(text.encode())
    return encrypted

# Decrypt text for displaying in dashboard
def decrypt_text(encrypted_text):
    key = load_key()
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_text)
    return decrypted.decode()