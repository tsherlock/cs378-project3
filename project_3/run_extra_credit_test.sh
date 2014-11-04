#!/usr/bin/env bash

# Run unit tests.
if python -m unittest -v extra_credit_test; then
	echo "All unit tests passing. Congrats!"
else
	echo "Some unit tests failing."
fi
