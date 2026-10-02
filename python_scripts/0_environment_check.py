import os
import platform
import sys


def main():
    print("=" * 60)
    print("LANGCHAIN COURSE ENVIRONMENT CHECK")
    print("=" * 60)

    print("\nHello, world!")
    print("Python is working successfully.")

    print("\nEnvironment details:")

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Operating system: {platform.platform()}")
    print(f"Current working directory: {os.getcwd()}")

    print("\nVirtual environment check:")

    if sys.prefix != sys.base_prefix:
        print("Virtual environment: Active")
        print(f"Environment location: {sys.prefix}")
    else:
        print("Virtual environment: Not active")

    print("\nBasic Python syntax check:")

    numbers = [10, 20, 30]
    total = sum(numbers)

    print(f"Numbers: {numbers}")
    print(f"Total: {total}")

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()