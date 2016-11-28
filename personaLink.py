#!/usr/bin/python



import sys

def main(argv):

	from personalinking import gen_pos_pairs, gen_all_neg_pairs, init
	import csv
	
	t = argv[0]
	d = argv[1]
	o = argv[2]



	if t=='train':

		gt = d+"/17_4_matches_training.json"

		f1 = d+"/usersFiltered17_training.json"
		f2 = d+"/usersFiltered14_training.json"
		f3 = d+"/postsFiltered17_training.json"
		f4 = d+"/postsFiltered14_training.json"
		init(f1, f2, f3, f4 )
		X_pos = gen_pos_pairs(gt)
		X_neg = gen_all_neg_pairs(gt, False)

		X = X_pos + X_neg

		p = len(X_pos)
		n = len(X_neg)

		f = open(o)
		writer = csv.writer(f)

		for i in range(p):

			writer.writerow(X_pos[i] + [1])
		for i in range(n):

			writer.writerow(X_neg[i] + [0])

		f.flush()
		f.close()

	if t=='test':


		gt = d+"/16_9_matches_j.txt"

		f1 = d+"/evalUsersAFiltered.txt"
		f2 = d+"/evalUsersBFiltered.txt"
		f3 = d+"/evalPostsAFiltered.txt"
		f4 = d+"/evalPostsBFiltered.txt"

		X = gen_all_neg_pairs(gt, True)

		f = open(o)
		writer = csv.writer(f)


		for item in X:

			writer.writerow(item)

		f.flush()
		f.close()


if __name__ == "__main__":
   main(sys.argv[1:])
