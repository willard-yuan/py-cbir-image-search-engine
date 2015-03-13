import pickle
from numpy import *
from imagesearch import imagesearch
from localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from tools.imtools import get_imlist

imlist = get_imlist('./first500/')
nbr_images = len(imlist)
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]


f = open('./vocabulary.pkl', 'rb')
voc = pickle.load(f)
f.close()

src = imagesearch.Searcher('web.db',voc)
locs,descr = sift.read_features_from_file(featlist[0])
iw = voc.project(array(descr))

print 'ask using a histogram...'
print src.candidates_from_histogram(iw)[:10]

src = imagesearch.Searcher('web.db',voc)
print 'try a query...'

nbr_results = 12
res = [w[1] for w in src.query(imlist[12])[:nbr_results]]
imagesearch.plot_results(src,res)