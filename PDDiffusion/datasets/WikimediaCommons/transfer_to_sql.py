"""Utility to transfer Wikimedia Commons data from the flat-file system to SQL."""

from argparse_dataclass import dataclass
from dataclasses import field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import sys

@dataclass
class TransferOptions:
    connection_string: str = field(metadata={"args": ["--url"], "help": "Connection string for the DB server or file to use"})

options = TransferOptions.parse_args(sys.argv[1:])
engine = create_engine(options.connection_string, future=True)

with Session(engine) as session:
    pass