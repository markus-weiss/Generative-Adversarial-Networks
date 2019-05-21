import os
import numpy as np
import fnmatch

class SignLanguageFiles(object):
    def __init__(self, database_path, search_string,signer_no, start):
        self.database_path = database_path
        self.search_string = search_string#e.g. 'iso' or 'con' or '*'
        self.signer_no = signer_no
        self.start = start
    #def __iter__(self):
        #return self
    def __iter__(self):
        signer_dirs = next( os.walk(self.database_path) )[1]
        #signer_dirs.sort()
        np.random.shuffle(signer_dirs)
        print(signer_dirs)

        for signer_dir in signer_dirs:#[self.signer_no:]:
            #print(signer_dir)
            signer_dir_full = os.path.join(self.database_path,signer_dir)
            signer_dir_full_flo = os.path.join(self.database_path+'_flowpngs',signer_dir)
            sign_dirs = next( os.walk (signer_dir_full)) [1]
            #sign_dirs.sort()
            #print('before shuffle: ', sign_dirs)
            np.random.shuffle(sign_dirs)
            #print('after shuffle: ', sign_dirs[1:4])
            #print('signer dirs: ', sign_dirs )
            for sign_dir in sign_dirs:#[self.start:]:
                #print(sign_dir)
                if sign_dir.find(self.search_string) > -1:
                    #print('sign_dir: ', sign_dir)
                    temp_label = sign_dir[-4:]
                    label = int(temp_label)
                    sign_dir_full = os.path.join(signer_dir_full, sign_dir)
                    sign_dir_full_flo = os.path.join(signer_dir_full_flo, sign_dir)
                    list_of_imgs = []
                    list_of_imgs_flo = []
                    all_files = next( os.walk(sign_dir_full))[2]
                    
                    #print(all_files)
                    files = fnmatch.filter(all_files, '*.jpg')
                    files.sort()
                    no_files = np.size(files)
                    #if (no_files != 80):
                        #print('sign_dir unequal 80 files', sign_dir, signer_dir)
                    m = 10
                    if no_files >= 350:
                        m = 50
                    elif no_files >= 280:
                        m = 30
                        
                    #print('no_files: ', no_files)
                    o_idx = no_files - m
                    a_idx = m
                    #print('matched files: ', files)
                    
#                    if no_files == 0:
#                       files = fnmatch.filter(all_files, '*.png')
#                       no_files = np.size(files) 
#                       o_idx = no_files - m +1
#                       assert(no_files > 0)
                    
                    for img in files[a_idx:o_idx]:
                        #print('img name: ', img)
                        img_flo = img[:-3]
                        img_flo = img_flo+'png'
                        img = os.path.join(sign_dir_full, img)
                        img_flo = os.path.join(sign_dir_full_flo, img_flo)
                        #print('img name full: ', img)
                        list_of_imgs.append(img)
                        list_of_imgs_flo.append(img_flo)
                    
                    translation_file = sign_dir_full + ".txt" 
                    
                    yield list_of_imgs, list_of_imgs_flo, translation_file, label