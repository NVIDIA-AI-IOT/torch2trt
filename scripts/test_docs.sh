#!/bin/bash

TAG=$1

<<<<<<< HEAD
mkdocs serve --dev-addr=0.0.0.0:8000
=======
mike set-default $TAG
mike serve --dev-addr=0.0.0.0:8000
>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06
