#!/usr/bin/python



import sys

def main(argv):

    from personalinking import gen_pos_pairs, gen_all_neg_pairs, init,gen_pair_pairs
    import csv,json
    
    t = argv[0]
    d = argv[1]
    o = argv[2]



    if t=='train':
        #proxy positives, found through exact and approximate username matches
        gt = d+'/pos_names.json'

        f1 = d+"/usersFiltered14_training.json"
        f2 = d+"/usersFiltered17_training.json"
        f3 = d+"/postsFiltered14_training.json"
        f4 = d+"/postsFiltered17_training.json"
        print 'init'
        init(f1, f2, f3, f4 )
        print 'gen pos pairs'
        pairdata = json.loads(open(gt).read())
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

if __name__ == "__main__":
   main(sys.argv[1:])
