"""Run with: python -m dashboard"""
from dashboard.app import create_app


def main(host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
