def checked_input(message = ""):
	user_line =""
	try:
		user_line = input(message+"\n")
	except EOFError:
		print("Invalid input try again...")
		checked_input(message)
	return user_line

