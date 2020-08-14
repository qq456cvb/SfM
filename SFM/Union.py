class UnionTree():
    def __init__(self, size):
        super().__init__()
        self.parents = list(range(size))
        self.ranks = [0] * size
        
    def find(self, a):
        while self.parents[a] != a:
            self.parents[a] = self.parents[self.parents[a]]
            a = self.parents[a]
        return a
    
    def union(self, a, b):
        a_root = self.find(a)
        b_root = self.find(b)
        
        if a_root == b_root:
            return
        
        if self.ranks[a_root] < self.ranks[b_root]:
            a_root, b_root = b_root, a_root
            
        self.parents[b_root] = a_root
        if self.ranks[a_root] == self.ranks[b_root]:
            self.ranks[a_root] += 1