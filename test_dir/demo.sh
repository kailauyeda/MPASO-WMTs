#!/bin/bash

# this is a hidden comment... yurrrrrrrrrrr
echo "Our script worked!"

# use the variable "FILE" to refer to a file name
FILE=nano_shortcuts.txt

# call wc - l on the file
wc -l $FILE

# call wc -l on the first argument listed after executing ./demo.sh filename1
wc -l $1

# run a for loop listing all of the files in this directory
#FILES=$(ls)
#for VAR in FILES
#do
#	echo $VAR
#done

for VAR in *.txt
do
	echo $VAR
done
