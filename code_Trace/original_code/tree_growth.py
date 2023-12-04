class Node(object):
    def __init__(self, count, value, level, children=None):
        self.count = count
        self.is_pruned = False
        self.children = children or []
        self.parent = None
        self.value = value
        self.level = level
    
    def add_child(self, obj):
        self.children.append(obj)
        obj.parent = self
    
    def prune(self):
        self.is_pruned = True

def node_update(root):
    children = root.children
    index = 0
    for node in children:
        index += 1
        node.count += index
    return root

if __name__ == '__main__':
    root = Node(0, None, 0)
    node1 = Node(0, 1, 1)
    node2 = Node(0, 2, 1)
    node3 = Node(0, 3, 1)
    root.add_child(node1)
    root.add_child(node2)
    root.add_child(node3)
    root = node_update(root)
    child = root.children
    for node in child:
        print(node.count)



    