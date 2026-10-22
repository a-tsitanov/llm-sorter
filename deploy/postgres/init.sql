-- база для temporal; расширение vector в базах sorter и sorter_test
CREATE DATABASE temporal;
CREATE DATABASE sorter_test;
\c sorter
CREATE EXTENSION IF NOT EXISTS vector;
\c sorter_test
CREATE EXTENSION IF NOT EXISTS vector;
