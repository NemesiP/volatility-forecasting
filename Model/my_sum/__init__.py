class calculate():
    def __init__(self, data):
        self.data = data
        
    def sum(self):
        total = 0
        for val in self.data:
            total += val
        return total