descr = []
descr.append(sift.read_features_from_file(featurefiles[0])[1])
descriptors = descr[0] #stack all features for k-means
print "start vstack descriptors"
for i in arange(1,nbr_images):
        descr.append(sift.read_features_from_file(featurefiles[i])[1])
        descriptors = vstack((descriptors,descr[i]))
            
# k-means: last number determines number of runs
print "start kmeans"
voc,distortion = kmeans(descriptors[::subsampling,:],k,1)
nbr_words = voc.shape[0]
        
# go through all training images and project on vocabulary
imwords = zeros((nbr_images, nbr_words))
for i in range( nbr_images ):
    imwords[i] = project(array(descr[i]))
        
    nbr_occurences = sum( (imwords > 0)*1 ,axis=0)
        
    idf = log( (1.0*nbr_images) / (1.0*nbr_occurences+1) )
    trainingdata = featurefiles

def project(descriptors):
    """ Project descriptors on the vocabulary
        to create a histogram of words. """
        
    # histogram of image words 
    imhist = zeros((nbr_words))
    words,distance = vq(descriptors,voc)
    for w in words:
        imhist[w] += 1
        
    return imhist