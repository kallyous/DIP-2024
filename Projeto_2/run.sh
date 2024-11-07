#!/bin/bash

# Verificar se o arquivo de nomes existe
if [ ! -f instances.txt ]; then
    echo "Arquivo instances.txt não encontrado!"
    exit 1
fi

# Ler o arquivo de texto linha por linha
while IFS= read -r inst; do
    # Rodar o script para cada ID
    pipenv run python 18211784.py --registration_number 12345 --input_filename "$inst"
done < instances.txt
