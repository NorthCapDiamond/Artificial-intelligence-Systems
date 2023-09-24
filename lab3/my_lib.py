def merge_underlined(array):
	if (len(array)==1):
		return array[0]
	else:
		answer = "_".join(array)
		return answer