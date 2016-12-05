import editdistance
from collections import Counter
import ujson
import ftfy
import random
import itertools
import numpy as np
import math
import string
import re

usrsA  = None
usrsB  = None
postsA = None
postsB = None

def init(f_usrsA, f_usrsB, f_postsA, f_postsB):

    global usrsA
    global usrsB
    global postsA
    global postsB
    global regex
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    usrsA = collapse_data(txt_dict(f_usrsA))
    preprocess_users(usrsA)
    usrsB = collapse_data(txt_dict(f_usrsB))
    preprocess_users(usrsB)
    postsA = usr_bow(txt_dict(f_postsA))
    postsB = usr_bow(txt_dict(f_postsB))
    preprocess_posts(postsA)
    preprocess_posts(postsB)

def txt_dict(filename):
     
    #test loading the file, printing sample contents
    with open(filename,'r') as fp:
        temp = ujson.load(fp);
    #print '# elements: ', len(temp1)
    #print 'sample of contents:'
    #print repr(temp1[1]).decode("unicode-escape")
    return temp

def preprocess_users(usrs):
    for key,item in usrs.iteritems():
        sig1 = item.get("Signature", None)
        if not sig1 is None:
            sig1 = replace_punct(sig1).lower().split()
            item['ssig'] = set(sig1)
            item['sig_tv'] = text_to_vector(sig1)
        im1 = getIM(item)
        item['getIM'] = im1

# In[2]:

def collapse_data(data):
    
    users = {}
    
    for item in data:
        if item[u'userName'] in users:
            user = users[item[u'userName']]
            if isinstance(item['groupName'],list):
                for name in item['groupName']:
                    if not name is None:
                        user['groupName'].add(name)
            else:
                user['groupName'].add(item['groupName'])    

            if item['imageName'] is not None:
                if isinstance(item['imageName'],list):
                    for l in zip(item['imageName'],item['imageHeight'],item['imageWidth'],item[u'imageMimeType'],item['imageSize']):
                        if not l[0] is None:
                            user['image'].add(l)
                else:
                    user['image'].add( (item['imageName'],item['imageHeight'],item['imageWidth'],item[u'imageMimeType'],item['imageSize']))
    
            fieldName = item.get('userProfileFieldName', None)
            
            if fieldName is not None:
            
                user[fieldName] = item['userProfileFieldValue']
    
        else:
        
            user = item['userName']
        
            users[user] = item.copy()
            if isinstance(item['groupName'],list):
                users[user]['groupName'] = set( [name for name in item['groupName'] if not name is None] )
            else:
                users[user]['groupName'] = set([item['groupName']])
            
            if isinstance(item['registrationTime'],list):
                users[user]['registrationTime']= min(item['registrationTime'])
            else:
                users[user]['registrationTime']= item['registrationTime']
            
            if item['imageName'] is not None:
                if isinstance(item['imageName'],list):
                    users[user]['image']= set( [l for l in zip(item['imageName'],item['imageHeight'],item['imageWidth'],item[u'imageMimeType'],item['imageSize']) if not l[0] is None])
                else:
                    users[user]['image']= set( [(item['imageName'],item['imageHeight'],item['imageWidth'],item[u'imageMimeType'],item['imageSize'])] ) 
            else:
                users[user]['image'] =set([])

            if isinstance(item['Signature'],list):
                users[user]['Signature'] = u' '.join(item['Signature'])
            else:
                users[user]['Signature'] = item['Signature']
        
    return users


# In[3]:

def usr_bow(tmp):
    
    if isinstance(tmp,list):
        posts = tmp
    else:
        posts = [ val for key,value in tmp.iteritems() for val in value]

    posts_bow = {}
   
    for post in posts:
    
        if posts_bow.get(post['userName'], None) == None:
            
            posts_bow[post['userName']] = {}
            
            posts_bow[post['userName']]['body'] = [post['postBody']  ]
            
            posts_bow[post['userName']]['subject'] = [post['postSubject']  ]
            
            posts_bow[post['userName']]['title'] = [post['threadTitle']  ]
            
            posts_bow[post['userName']]['postTime'] = [post['postTime']]
        
         
        else:
            
            posts_bow[post['userName']]['body'] = posts_bow[post['userName']]['body']  + [post['postBody']]
            
            posts_bow[post['userName']]['subject'] = posts_bow[post['userName']]['subject'] +[post['postSubject']  ]

            posts_bow[post['userName']]['title'] = posts_bow[post['userName']]['title'] + [post['threadTitle']  ]
            
            posts_bow[post['userName']]['postTime'] = posts_bow[post['userName']]['postTime'] + [post['postTime']]
            
    return posts_bow


# In[ ]:

def check_exist(usrA, usrB, usrsA, usrsB, postsA, postsB, ):
    
    
    if usrA not in usrsA.keys():
        
        return 0
    
    if usrB not in usrsB.keys():
        
        return 0
    
    if usrA not in postsA.keys():
        
        return 0
    
    if usrB not in postsB.keys():
        
        return 0
    
    return 1


# In[4]:

def gen_pair_pairs(data_gt,keya='site_a',keyb='site_b'):

    global usrsA
    global usrsB
    global postsA
    global postsB

    X = []
    
    for pair in data_gt:
            usr1 = pair[keya]
            
            usr2 = pair[keyb]
            
            X.append(featurise((usr1, usr2) ))
    return X


def gen_pos_pairs(data_gt,keya='site_a',keyb='site_b'):

    global usrsA
    global usrsB
    global postsA
    global postsB

    X = []
    
    for pair in data_gt:
        
        if pair['class'] == 1:
            
            usr1 = pair[keya]
            
            usr2 = pair[keyb]
            
           # if not check_exist(usr1, usr2, usrsA, usrsB, postsA, postsB):
                
            #    continue
            if usr2 == u'Ghost\u2122':
                usr2 = "GhostTM"
            if usr2 == u'nicklan&lt;b&gt;&lt;/b&gt;':
                usr2 = u'nicklan<b></b>'
            
            #X.append(featurise(usr1, usr2,  postsA, postsB, usrsA, usrsB))
            X.append(featurise((usr1, usr2) ))
                
    return X


# In[5]:

def gen_neg_pairs(filename,keya='site_a',keyb='site_b'):

    global usrsA
    global usrsB
    global postsA
    global postsB
    
    
    with open(filename, 'rb') as f:
        data = f.readlines()

    data = map(lambda x: x.rstrip(), data)

    # convert to array of JSON objects
    data_json_str = "[" + ','.join(data) + "]"

    # load it
    data_gt = ujson.loads(data_json_str)

    X = []
    
    pos = []
    

    
    for pair in data_gt:
        
        if pair['class'] == 1:
            
            pos.append((pair[keya],pair[keyb]))
            
    for i in range(len(data_gt)):
        
        
        usr1 = random.choice(usrsA.keys())
        
        usr2 = random.choice(usrsB.keys())
        
        if (usr1, usr2) not in pos:
        
        
            #X.append(featurise(usr1, usr2,  postsA, postsB, usrsA, usrsB))
            X.append(featurise((usr1, usr2)))
            
    return X




def gen_all_neg_pairs(data_gt, evl=False,keya='site_a',keyb='site_b'):

    global usrsA
    global usrsB
    global postsA
    global postsB
    
    if evl:

        pos = []

    else:

        X = []
    
        pos = []
        
        for pair in data_gt:
        
            if pair['class'] == 1:
            
                pos.append((pair[keya],pair[keyb]))
            
    targets = []
    print "HERE GOES NOTHING"        
        
            
    for usr1 in usrsA.keys():
        
        for usr2 in usrsB.keys():
        
        
        #usr1 = random.choice(usrsA.keys())
        
        #usr2 = random.choice(usrsB.keys())
        
            if (usr1, usr2) not in pos:
        
                targets.append((usr1, usr2))
    print "HERE GOES NOTHING AGAIN"        
    print len(targets)        
    from multiprocessing import Pool
    
    p = Pool(40)
    
    X = p.map(featurise, targets)#, chunksize=10000 )
    
    p.close()
    p.join()
    
    return X
            
                         
                         
def text_to_vector(text):
    return Counter(text)


# In[7]:

def get_cosine(vec1, vec2):
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([x**2 for x in vec1.values()])
    sum2 = sum([x**2 for x in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def replace_punct(st):
    

    
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', st)
    
    return out

def preprocess_posts(postdict):
    for key,item in postdict.iteritems():
        item['bodyset'] = [ set(replace_punct(x).lower().split()) for x in item['body']]
        item['subjectset'] = [ set(replace_punct(x).lower().split()) for x in item['subject']]
        item['titleset'] = [ set(replace_punct(x).lower().split()) for x in item['title']]
        item['bodyvec'] = [ text_to_vector(replace_punct(x).lower().split()) for x in item['body']]
        item['subjectvec'] = [ text_to_vector(replace_punct(x).lower().split()) for x in item['subject']]
        item['titlevec'] = [ text_to_vector(replace_punct(x).lower().split()) for x in item['title']]

        item['bodyset_compl'] = frozenset().union(*item['bodyset'])
        item['subjectset_compl'] = frozenset().union(*item['subjectset'])
        item['titleset_compl'] = frozenset().union(*item['titleset'])

        item['bodyvec_compl'] = text_to_vector(list(itertools.chain(*[replace_punct(x).lower().split() for x in item['body']])))
        item['subjectvec_compl'] = text_to_vector(list(itertools.chain(*[replace_punct(x).lower().split() for x in item['subject']])))
        item['titlevec_compl'] = text_to_vector(list(itertools.chain(*[replace_punct(x).lower().split() for x in item['title']])))
        
def gen_jac(l1,l2):
    for item in l1:
        for item2 in l2:
            if len(item|item2) == 0:
                yield 0.0
            else:
                 yield float(len(item&item2))/(len(item|item2)) 

def gen_cos(v1,v2):
    for vec1 in v1:
        for vec2 in v2:
            yield get_cosine(vec1, vec2)

def indi_scores(l1, l2, v1, v2):
    
    scores_jac = np.array([x for x in gen_jac(l1,l2)])
    
    scores_cs = np.array([ x for x in gen_cos(v1,v2)])
    
    cs_feats = [ scores_cs.min(), scores_cs.max(), scores_cs.mean() ]
    
    jac_feats = [scores_jac.min(), scores_jac.max(), scores_jac.mean()]
    
    return cs_feats + jac_feats 


# In[9]:




def overall_scores(item, item2,v1,v2):
    
     
    if len(item|item2) == 0:
            
        s1 = 0.0
        
    else:
        s1 = float(len(item&item2))/(len(item|item2)) 

    
    s2 = get_cosine(v1, v2)
        
    return s1,s2

def post_freq(time):
    
    mn = min(time)
    
    mx = max(time)
    
    return float(mx-mn)/len(time)

def post_std(time):
    
    return np.std(time)

def post_mn(time):
    
    return np.mean(time)

def post_time(usr1, usr2, postsA, postsB, usrsA, usrsB):
    
    time1 = [int(post) for post in postsA[usr1]['postTime'] ]
    
    time2 = [int(post) for post in postsB[usr2]['postTime'] ]
                  
    return abs(post_std(time1) - post_std(time2 ) ),abs(post_freq(time1) - post_freq(time2) ),abs( post_mn(time1) - post_mn(time2)  )
    
    

def feature_gen(usrs):
        global usrsA
        global usrsB
        global postsA
        global postsB
    
        usr1 = usrs[0]
        usr2 = usrs[1]
        yield usr1.encode('utf-8')
        yield usr2.encode('utf-8')
        #feature1 : editDist between usernames
        username1 = usr1
        username2 = usr2
        #ft1 = editdistance.eval(usr1, usr2)
        #yield int(ft1)

        #feature2 : editDist between lowercase usernames
        #ft2 = editdistance.eval(usr1.lower(), usr2.lower())
        #yield int(ft2)

        #if len(usr1) > len(usr2):
        #        norm = len(usr1)
        #else:
        #        norm = len(usr2)

        #ft1 = float(ft1)/norm
        #ft2 = float(ft2)/norm

        #yield ft1
        #yield ft2

        #feature3 : timediff between user     
        yield int(abs(usrsA[usr1]['registrationTime'] - usrsB[usr2]['registrationTime']))#0

        #feature4: TimeZone
        if usrsA[usr1]['Time Zone'] == usrsB[usr2]['Time Zone']:
            yield 1#1
        else:
            yield 0

        #text Features

        #feature 5: Textual, Individual, overall

        if (postsA.get(usr1, None) != None) and (postsB.get(usr2, None) != None):

            for sc in indi_scores(postsA[usr1]['bodyset'],postsB[usr2]['bodyset'],postsA[usr1]['bodyvec'],postsB[usr2]['bodyvec']):
                yield sc#2,3,4,5,6,7

            for sc in indi_scores(postsA[usr1]['subjectset'],postsB[usr2]['subjectset'],postsA[usr1]['subjectvec'],postsB[usr2]['subjectvec']):
                yield sc#8,9,10,11,12,13

            for sc in indi_scores(postsA[usr1]['titleset'],postsB[usr2]['titleset'],postsA[usr1]['titlevec'],postsB[usr2]['titlevec']):
                yield sc#14,15,16,17,18,19

            for sc in overall_scores(postsA[usr1]['bodyset_compl'],postsB[usr2]['bodyset_compl'],postsA[usr1]['bodyvec_compl'],postsB[usr2]['bodyvec_compl']):
                yield sc#20,21

            for sc in overall_scores(postsA[usr1]['subjectset_compl'],postsB[usr2]['subjectset_compl'],postsA[usr1]['subjectvec_compl'],postsB[usr2]['subjectvec_compl']):
                yield sc#22,23

            for sc in overall_scores(postsA[usr1]['titleset_compl'],postsB[usr2]['titleset_compl'],postsA[usr1]['titlevec_compl'],postsB[usr2]['titlevec_compl']):
                yield sc#24,25

            for sc in post_time( usr1, usr2, postsA, postsB, usrsA, usrsB ):
                yield sc#26,27,28

        else:
            for sc in xrange(27):
                yield -1

        #feature7: Group Affliations

        groups1 = usrsA[usr1]['groupName']

        groups2 = usrsB[usr2]['groupName']

        n = 0

        for item in groups1:

            for item2 in groups2:

                if item[:3].lower() == item2[:3].lower():

                    n+=1

        yield n#29

        #feature8: Signatures

        ft8 = -1

        ft9 = -1

        if ('ssig' in usrsA[usr1]) and ('ssig' in usrsB[usr2]):
            ssig1 = usrsA[usr1]['ssig']

            ssig2 = usrsB[usr2]['ssig']

            ft8 = float(len(ssig1&ssig2))/float((len(ssig1|ssig2)))

            ft9 = get_cosine(usrsA[usr1]['sig_tv'], usrsB[usr2]['sig_tv'])

        yield ft8#30
        yield ft9#31

        #feature9: image
        
        image1 = usrsA[usr1].get('image')
        image2 = usrsB[usr2].get('image')
        
        ft10 = -1
        ft11 = -1
        if (len(image1) > 0 ) and ( len(image2)>0  ):
                
            image1 = set([img[0] for img in image1   ])
            image2 = set([img[0] for img in image2   ])
            
            ft10 = len(image1&image2  )
            ft11 = len(image1^image2  )
        
        yield ft10#32
        yield ft11#33
        
        #feature10: IM
        
        im1 = usrsA[usr1]['getIM']
        im2 = usrsB[usr2]['getIM']

        yield len(im1&im2)
        yield len(im1^im2)

        
def featurise(usrs):
        return [x for x in feature_gen(usrs)]

def getIM(ft):
    l = [ft[x] for x in ['AOL IM',u'Jabber','ICQ','Yahoo! IM','MSN IM','Skype'] if x in ft] 
    return set(list(itertools.chain(*[x for x in l if not x is None])))
        
def check_feat(key, d):
    
    for item in d:
        
        if key not in d[item].keys():
            return "Not There", item
    
    
    return "there in every element! :)"

