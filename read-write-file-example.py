file1 = open("myfile.txt", "w")
L = ["This is Delhi \n", "This is Paris \n", "This is London \n"]

# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close()  # to change file access modes

file1 = open("myfile.txt", "r+")

print("1 Output of Read function is ")
print(file1.read())
print()

# seek(n) takes the file handle to the nth
# byte from the beginning.

file1.seek(0)
print("2 Output of Readline function is ")
print(file1.readline())
print()

file1.seek(0)

# To show difference between read and readline
print("3 Output of Read(9) function is ")
print(file1.read(9))
print()

file1.seek(0)

print("4 Output of Readline(9) function is ")
print(file1.readline(9))

file1.seek(0)
# readlines function
print("5 Output of Readlines function is ")
print(file1.readlines())
print()
file1.close()
