#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from django.test import Client, TestCase


class IndexTestCase(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_get_index_endpoint(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'Hello, world')


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'set.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
