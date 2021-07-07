import torch.nn as nn

class PenetrationModel:
  def __init__(self, hand_model, object_model):
    self.hand_model = hand_model
    self.object_model = object_model
    self.relu = nn.ReLU()

  def get_penetration(self, object_code, hand_code):
    hand_vertices = self.hand_model.get_vertices(hand_code)
    return self.get_penetration_from_verts(object_code, hand_vertices)

  def get_penetration_from_verts(self, object_code, hand_vertices):
    h2o_distances = self.object_model.distance(object_code, hand_vertices) # B x V x 1
    penetration = self.relu(h2o_distances).squeeze(-1)
    return penetration
