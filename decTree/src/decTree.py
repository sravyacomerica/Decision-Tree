#!/usr/bin/env python

class decTree:
    def __init__(self, text, left=None, right=None):
        self.txt = str(text) ## The name of the attribute or the prediction for that leaf
        self.l = left        ## Left subtree, corresponding to  txt==0
        self.r = right       ## Right subtree, corresponding to txt==1 

    def insert(self, data):
        if self.value == data:
            return False

    ## printing method
    def __str__(self):
        return self.toString(0)

    def toString(self, depth=0):
        if self.isLeaf(): return self.txt

        extra_l = ""
        if not self.l.isLeaf(): extra_l = "\n"
        extra_r = ""
        if not self.r.isLeaf(): extra_r = "\n"

        return depth*"| " + self.txt + " = 0 : " + extra_l +\
            self.l.toString(depth+1) + "\n" +\
            depth*"| " + self.txt + " = 1 : " + extra_r +\
            self.r.toString(depth+1)

    def isLeaf(self):
        ## A tree is a leaf if it doesn't have daughters
        return self.l is None and self.r is None

if __name__ == "__main__":

    t = decTree('attr1',
                 decTree('attr2', decTree('attr3', decTree('0'), decTree('1')), decTree('0')),
                 decTree('attr3', decTree('0'), decTree('1')),
                 )

    print (t)