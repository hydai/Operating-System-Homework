for (( i = 1; i < 5; i++ )); do
    gcc -o p${i} p${i}.c
done
