class CentroidAssignmentStragety:
    def __init__(self, item_code_bytes: int, num_items: int) -> None:
        self.item_code_bytes = item_code_bytes
        self.num_items = num_items

    def pass_item_mappings(self, item_mappings) -> None:
        del item_mappings

    def assign(self, train_users):
        raise NotImplementedError
