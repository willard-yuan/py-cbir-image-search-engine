import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

imlist = get_imlist('./first500/')
nbr_images = len(imlist)
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]


f = open('./first500/vocabulary.pkl', 'rb')
voc = pickle.load(f)
f.close()

src = imagesearch.Searcher('web.db',voc)
locs,descr = sift.read_features_from_file(featlist[0])
iw = voc.project(descr)

print 'ask using a histogram...'
print src.candidates_from_histogram(iw)[:10]

src = imagesearch.Searcher('web.db',voc)
print 'try a query...'

nbr_results = 12
res = [w[1] for w in src.query(imlist[39])[:nbr_results]]
imagesearch.plot_results(src,res)