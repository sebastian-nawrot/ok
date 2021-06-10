import datetime
import matplotlib.pyplot
import networkx
import pathlib
import random

from collections import Counter
from copy import copy
from networkx import all_simple_paths as find_paths
from networkx.generators.degree_seq import random_degree_sequence_graph


regress_duration = 8
tabu_duration = 11
config_neighbours = 200
time = datetime.timedelta(seconds=20)


class Solution:
  def __init__(self, instance, path: list[int]):
    self.instance = instance
    self.path = path
    self.cost = self.calculate_cost(self.path)
    self.tabu = [0] * len(self.path)
    self.without_repeats = False

  def improve(self):
    self.tabu = [value - 1 if value > 1 else 0 for value in self.tabu]
    neighbours = sorted(self.find_neighbours(self.path))
    for cost, neighbour, (left, right, left_insert, right_insert) in neighbours:
      if cost >= self.cost:
        return False

      self.tabu[left] = self.tabu[right] = tabu_duration
      self.tabu = (self.tabu[:left] + [0] * left_insert +
                   self.tabu[left:right+1][::-1] + [0] * right_insert +
                   self.tabu[right+1:])
      self.path, self.cost = neighbour, cost

      if not self.without_repeats:
        counter = {k: v for k, v in Counter(self.path).items() if v > 1}
        if not len(counter):
          self.without_repeats = True
        while self.remove_repeats(counter):
          pass
      return True
    else:
      return False

  def regress(self):
    self.tabu = [value - 1 if value > 1 else 0 for value in self.tabu]
    previous_cost = self.cost
    neighbours = sorted(self.find_neighbours(self.path))
    for cost, neighbour, (left, right, left_insert, right_insert) in neighbours:
      if cost > previous_cost:
        self.tabu[left] = self.tabu[right] = tabu_duration
        self.tabu = (self.tabu[:left] + [0] * left_insert +
                    self.tabu[left:right+1][::-1] + [0] * right_insert +
                    self.tabu[right+1:])

        self.path, self.cost = neighbour, cost
        return True
    else:
      return False


  def find_neighbours(self, path):
    graph = self.instance.graph
    index, maximum = 0, config_neighbours
    path_len = len(path)

    # 2-opt
    for i in range(0, path_len - 1):
      if self.tabu[i]: continue
      for j in range(i + 1, path_len):
        if self.tabu[j]: continue

        #if index >= maximum:
        #  return

        if i > 0:
          if path[j] not in graph[path[i - 1]]:
            continue

        if j < path_len - 1:
          if path[i] not in graph[path[j + 1]]:
            continue

        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
        yield self.calculate_cost(new_path), new_path, (i, j, 0, 0)
        index += 1

    while index < maximum:
      i, j = sorted(random.sample(range(path_len), 2))

      left_part, left_insert = path[:i], 0
      if i > 0:
        if path[j] not in graph[path[i - 1]]:
          for inner in find_paths(graph, path[i-1], path[j], 3):
            if inner[:-1] != path[i:j+1]:
              left_part, left_insert = left_part + inner[1:-1], len(inner[1:-1])
              break
          else:
            continue

      right_part, right_insert = path[j+1:], 0
      if j < path_len - 1:
        if path[i] not in graph[path[j + 1]]:
          for inner in find_paths(graph, path[i], path[j+1], 3):
            if inner[:-1] != path[i:j+1]:
              right_part, right_insert = inner[1:-1] + right_part, len(inner[1:-1])
              break
          else:
            continue

      new_path = left_part + path[i:j+1][::-1] + right_part
      yield (self.calculate_cost(new_path), new_path, (i, j, left_insert, right_insert))
      index += 1


  def remove_repeats(self, counter):
    graph = self.instance.graph
    path = self.path

    for node, occurances in counter.items():
      if occurances > 1:
        last_index = 0
        for _ in range(occurances):
          index = path.index(node, last_index)
          last_index = index + 1

          if index > 0 and index < len(path) - 1:
            if path[index - 1] not in graph[path[index + 1]]:
              continue

          new_path = copy(path)
          new_path.pop(index)
          new_path_cost = self.calculate_cost(path)

          if self.tabu[index]:
            continue

          self.cost = new_path_cost
          self.path.pop(index)
          self.tabu.pop(index)

          counter[node] -= 1
          return True
    return False


  def calculate_cost(self, path: list[int]):
    edges = self.instance.graph.edges
    graph = self.instance.graph
    path_len = len(path) - 1

    weights = [edges[path[i], path[i + 1]]['weight'] for i in range(0, path_len)]
    weights_len = len(weights)

    weight = 0
    for i in range(5, weights_len, 5):
      weight += sum(weights[i-5:i]) + sum(weights[i-3:i]) * len(graph[path[i]]) * 2
    return weight + sum(weights[i:])



class Instance:
  def __init__(self, pool, name, graph: networkx.Graph):
    self.pool = pool
    self.name = name
    self.graph = graph
    self.pool.render_graph(self)
    self.first_solution = Solution(self, self.find_first_solution())
    self.best_solution = copy(self.first_solution)


  def tabu_search(self):
    self.values = [self.first_solution.cost]
    self.best_values = [self.first_solution.cost]
    self.timestamps = [0]

    start = datetime.datetime.now()
    stop = datetime.datetime.now() + time
    current = copy(self.first_solution)
    while datetime.datetime.now() < stop:
      while current.improve():
        if current.cost < self.best_solution.cost:
          print(current.cost)
          self.best_solution.cost = current.cost
          self.best_solution = copy(current)
        self.timestamps.append((datetime.datetime.now() - start).seconds)
        self.values.append(current.cost)
        self.best_values.append(self.best_solution.cost)
      else:
        for _ in range(regress_duration):
          current.regress()


  def find_first_solution(self):
    current = 0
    solution = [list(self.graph.nodes())[0]]
    while set(solution) != set(self.graph.nodes):
      assert current >= 0
      for adjacent in sorted(self.graph[solution[current]]):
        if adjacent not in solution:
          if current < len(solution) - 1:
            solution.insert(current + 1, solution[current])
          solution.insert(current + 1, adjacent)
          current += 1
          break
      else:
        current -= 1
    return solution



class InstancePool:
  def __init__(self, instance_size: int, vertices_amount: int, min_edges: int,
               max_edges: int, min_weight: int, max_weight: int, name: str = None):
    self.instance_size = instance_size
    self.vertices_amount = vertices_amount
    self.min_edges = min_edges
    self.max_edges = max_edges
    self.min_weight = min_weight
    self.max_weight = max_weight

    if name:
      self.pool_name = f'{name}__{instance_size}_{vertices_amount}_{min_edges}_' \
                       f'{max_edges}_{min_weight}_{max_weight}_{regress_duration}_' \
                       f'{tabu_duration}_{config_neighbours}'
    else:
      self.pool_name = f'{instance_size}_{vertices_amount}_{min_edges}_{max_edges}_' \
                       f'{min_weight}_{max_weight}_{regress_duration}_{tabu_duration}_' \
                       f'{config_neighbours}'

    self.directory = pathlib.Path('instances/' + self.pool_name)
    self.directory.mkdir(parents=True, exist_ok=True)

    self.instances = []

    
  def generate_graphs(self):
    for index in range(self.instance_size):
      self.instances.append(Instance(self, f'instance_{index:02}', self._generate_graph(index)))


  def load_graphs(self):
    for index, each in enumerate(self.directory.glob('instance_*')):
      graph = networkx.read_weighted_edgelist(each)
      mapping = {each: int(each) for each in graph.nodes}
      graph = networkx.relabel_nodes(graph, mapping)
      for edge in graph.edges:
        graph.edges[edge]['weight'] = int(graph.edges[edge]['weight'])
      self.instances.append(Instance(self, f'instance_{index:02}', graph))


  def tabu_search(self):
    print('running')
    for each in self.instances:
      each.tabu_search()

      reduction = each.first_solution.cost - each.best_solution.cost
      reduction_percent = each.best_solution.cost / each.first_solution.cost * 100
      print(f'{each.first_solution.cost:6} - {reduction} = {each.best_solution.cost:6} {int(reduction_percent)}')

      self.make_plot(each)

    self.dump_values()


  def make_plot(self, instance, custom_name=None):
    matplotlib.pyplot.clf()
    delta = instance.first_solution.cost - instance.best_solution.cost
    reducted_by = int(delta / instance.first_solution.cost * 100)
    title = f'{instance.name}, wartość początkowa: {instance.first_solution.cost}, ' \
            f'wartość końcowa: {instance.best_solution.cost}, zredukowano o {reducted_by}%\n'
    title += f'ilość wierzchołków: {self.vertices_amount}, min ilość krawędzi: {self.min_edges}, ' \
             f'max ilość krawędzi: {self.max_edges}, min waga krawędzi: {self.min_weight}, ' \
             f'max waga krawędzi: {self.max_weight}\n'
    title += f'ilość iteracji pogarszających: {regress_duration}, blokada ruchu: {tabu_duration}, ' \
             f'ilość sąsiedztw: {config_neighbours}, czas wykonywania: {time}'

    matplotlib.pyplot.title(title, fontsize=10)

    matplotlib.pyplot.plot(list(range(1, len(instance.values) + 1)), instance.best_values, label='Najlepsza wartość')
    matplotlib.pyplot.plot(list(range(1, len(instance.values) + 1)), instance.values, label='Obecna wartość')

    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Numer iteracji')
    matplotlib.pyplot.ylabel('Wartość funkcji celu')

    matplotlib.pyplot.gcf().set_size_inches(10, 10)

    if not custom_name:
      matplotlib.pyplot.savefig(str(self.directory / f'wykres_{instance.name}.png'))
    else:
      matplotlib.pyplot.savefig(str(self.directory / f'{custom_name}.png'))


  def dump_values(self):
    with open(str(self.directory / f'best_values.txt'), 'w') as file:
      for each in self.instances:
        assert len(each.timestamps) == len(each.best_values)
        file.write('[')
        for timestamp, value in zip(each.timestamps, each.best_values):
          file.write(f'({timestamp}, {value}), ')
        file.write(']\n')


  def render_graph(self, instance):
    matplotlib.pyplot.clf()

    networkx.set_edge_attributes(instance.graph, 'b', 'color')
    networkx.draw(instance.graph, with_labels=True, font_weight='bold', node_color='lightgreen')

    matplotlib.pyplot.gcf().set_size_inches(35, 35)
    matplotlib.pyplot.savefig(str(self.directory / f'graph_{instance.name}.png'))


  def _generate_graph(self, index):
    while True:
      try:
        degree_sequence = sorted(random.choices(range(self.min_edges, self.max_edges), k=self.vertices_amount))
        graph = random_degree_sequence_graph(degree_sequence, tries=200)

        weights = {(x, y): random.randint(self.min_weight, self.max_weight) for x, y in graph.edges}
        networkx.set_edge_attributes(graph, weights, 'weight')
        networkx.write_weighted_edgelist(graph, str(self.directory / f'instance_{index:02}'))
        return graph
      except networkx.exception.NetworkXUnfeasible as exception:
        pass



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Tabu search')
  parser.add_argument('name', type=str)

  parser.add_argument('instances_number', type=int)
  parser.add_argument('vertices_number', type=int)
  parser.add_argument('min_edges', type=int)
  parser.add_argument('max_edges', type=int)
  parser.add_argument('min_weight', type=int)
  parser.add_argument('max_weight', type=int)

  parser.add_argument('regress_duration', type=int)
  parser.add_argument('tabu_duration', type=int)
  parser.add_argument('neighbours', type=int)
  parser.add_argument('time', type=int)

  parser.add_argument('--use_existing', action='store_true')

  args = parser.parse_args()


  regress_duration = args.regress_duration
  tabu_duration = args.tabu_duration
  config_neighbours = args.neighbours
  time = datetime.timedelta(seconds=args.time)

  pool = InstancePool(args.instances_number, args.vertices_number,
                      args.min_edges, args.max_edges,
                      args.min_weight, args.max_weight, args.name)
  if args.use_existing:
    pool.load_graphs()
  else:
    pool.generate_graphs()
  pool.tabu_search()