#!/usr/bin/python



import sys

def main(argv):

	from personalinking import gen_pos_pairs, gen_all_neg_pairs, init
	import csv,json
	
	t = argv[0]
	d = argv[1]
	o = argv[2]



	if t=='train':
		gt = d+"/17_14_matches_training.json"
		gt2 = d+"/17_14_matches_training2.json"

		f1 = d+"/usersFiltered14_training.json"
		f2 = d+"/usersFiltered17_training.json"
		f3 = d+"/postsFiltered14_training.json"
		f4 = d+"/postsFiltered17_training.json"
		print 'init'
		init(f1, f2, f3, f4 )
		print 'gen pos pairs'
		pairdata = json.loads(open(gt).read()) + json.loads(open(gt2).read())
		X_pos = gen_pos_pairs(pairdata,keya='site_14',keyb='site_17')
		print 'gen neg pairs'
		X_neg = gen_all_neg_pairs(pairdata, False,keya='site_14',keyb='site_17')

		f = open(o,'w')
		writer = csv.writer(f)

		for row in X_pos:
			writer.writerow(row + [1])
		for row in X_neg:
			writer.writerow(row + [0])

		f.flush()
		f.close()

	if t=='test':


		gt = d+"/16_9_matches_j.txt"

		f1 = d+"/evalUsersAFiltered.txt"
		f2 = d+"/evalUsersBFiltered.txt"
		f3 = d+"/evalPostsAFiltered.txt"
		f4 = d+"/evalPostsBFiltered.txt"

		X = gen_all_neg_pairs(gt, True)

		f = open(o,'w')
		writer = csv.writer(f)


		for item in X:

			writer.writerow(item)

		f.flush()
		f.close()


if __name__ == "__main__":
   main(sys.argv[1:])
