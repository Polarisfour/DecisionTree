import math
import copy

# Example class for train and test data
class Example:
    def __init__(self, features, label) -> None:
        # Map of attribute to value
        self.features = features
        # Classification of example
        self.label = label

# Node class for decision tree
class Node:
    def __init__(self, label=False, attribute=None, children=None, classification=None, values=None) -> None:
        # Classification label (str)
        self.classification = classification
        # Whether it is a label (bool)
        self.label = label
        # The arribute of the node (str)
        self.attribute = attribute
        # Values of the attribute (list)
        self.values = values
        # Children of the node (map of values to Nodes)
        self.children = children


# Attribute class to hold values
class Attribute:
    def __init__(self, name, values) -> None:
        # Name of attribute (str)
        self.name = name
        # Attribute values (list)
        self.values = values


class DecisionTree:
    def __init__(self, importanceMethod) -> None:
        self.importance = importanceMethod

    # Determines the classification with the highest presence in examples
    def plurality_value(self, examples):
        values = {}
        for example in examples:
            if example.label in values.keys():
                values[example.label] = values[example.label] + 1
            else:
                values[example.label] = 1
                max_key = example.label
        keys = values.keys()
        for key in keys:
            if values[key] > values[max_key]:
                max_key = key
        return max_key

    # Determines whether all examples have the same classification
    def same_classification(self, examples):
        classification = examples[0].label
        for example in examples:
            if example.label != classification:
                return False
        return True

    # Give a list of all of the remaining classifications in examples
    def list_labels(self, examples):
        labels = []
        for example in examples:
            if example.label in labels:
                continue
            else:
                labels.append(example.label)
        return labels

    # Find the percentage (as a decimal) of examples with a given label
    def percent_labeled(self, examples, label):
        sum = 0.0
        for example in examples:
            if example.label == label:
                sum = sum + 1.0
        percent = sum / float(len(examples))
        return percent

    # Determines the entropy of a set of examples
    def entropy(self, examples):
        entropy = 0.0
        labels = self.list_labels(examples)
        for label in labels:
            percent = self.percent_labeled(examples, label)
            if percent == 0:
                log = 0
            else:
                log = math.log2(percent)
            temp = percent * log
            entropy = entropy + temp
        entropy = entropy * -1
        return entropy

    # Determines the information gain with the selection of a certain example

    def information_gain(self, examples, attribute):
        total_entropy = self.entropy(examples)
        sum = 0.0
        for value in attribute.values:
            value_examples = self.split_examples(examples, attribute.name, value)
            ratio = float(len(value_examples)) / float(len(examples))
            value_entropy = self.entropy(value_examples)
            temp = ratio * value_entropy
            sum += temp
        information_gain = total_entropy - sum
        return information_gain

    # Finds the intrinsic value of a set of examples with a certain attribute
    def intrinsic_value(self, examples, attribute):
        sum = 0.0
        total_count = float(len(examples))
        for value in attribute.values:
            value_examples = self.split_examples(examples, attribute.name, value)
            value_count = float(len(value_examples))
            ratio = value_count / total_count
            if ratio == 0.0:
                log = 0.0
            else:
                log = math.log2(ratio)
            temp = ratio * log
            sum = sum + temp
        intrinsic_value = -1 * sum
        if intrinsic_value == 0:
            intrinsic_value = 0.00001
        return intrinsic_value

    # Determines the gain ration when selecting a specific attribute
    def gain_ratio(self, examples, attribute):
        information_gain = self.information_gain(examples, attribute)
        intrinsic_value = self.intrinsic_value(examples, attribute)
        gain_ratio = information_gain / intrinsic_value
        return gain_ratio

    # Determines the Gini value of a set of examples
    def gini_value(self, examples):
        labels = self.list_labels(examples)
        sum = 0.0
        for label in labels:
            percent_labeled = self.percent_labeled(examples, label)
            percent_squared = percent_labeled * percent_labeled
            sum = sum + percent_squared
        gini_value = 1 - sum
        return gini_value

    # Determines the gini index of a set of examples given an attribute
    def gini_index(self, examples, attribute):
        sum = 0.0
        total_count = float(len(examples))
        for value in attribute.values:
            value_examples = self.split_examples(examples, attribute.name, value)
            value_count = float(len(value_examples))
            ratio = value_count / total_count
            gini_value = self.gini_value(value_examples)
            temp = ratio * gini_value
            sum = sum + temp
        gini_index = sum
        return gini_index

    # Determines the attribute that has the highest importance to select next
    def highest_importance(self, attributes, examples):
        if self.importance == 0:
            importance_map = {}
            for attribute in attributes:
                importance_map[attribute] = self.gain_ratio(examples, attribute)
            max_attribute = attributes[0]
            max_gain = importance_map[max_attribute]
            for attribute in attributes:
                if importance_map[attribute] > max_gain:
                    max_gain = importance_map[attribute]
                    max_attribute = attribute
            return max_attribute
        else:
            importance_map = {}
            for attribute in attributes:
                importance_map[attribute] = self.gini_index(examples, attribute)
            min_attribute = attributes[0]
            min_index = importance_map[min_attribute]
            for attrbiute in attributes:
                if importance_map[attribute] < min_index:
                    min_attribute = attribute
                    min_index = importance_map[attribute]
            return min_attribute

    # Split the examples into a new list that contains examples with an attribute at a specified value
    def split_examples(self, examples, attribute, value):
        new_examples = []
        for example in examples:
            if example.features[attribute] == value:
                new_examples.append(example)
        return new_examples

    # Recursive tree creation function
    def create_tree(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return Node(label=True, classification=self.plurality_value(parent_examples))
        elif self.same_classification(examples):
            return Node(label=True, classification=examples[0].label)
        elif len(attributes) == 0:
            return Node(label=True, classification=self.plurality_value(examples))
        else:
            A = self.highest_importance(attributes, examples)
            tree = Node(attribute=A.name, children={}, values=A.values)
            for value in tree.values:
                sub_examples = self.split_examples(examples, tree.attribute, value)
                new_attributes = attributes.copy()
                new_attributes.remove(A)
                tree.children[value] = self.create_tree(sub_examples, new_attributes, examples)
            return tree

    # Associates the tree with the object
    def add_tree(self, tree):
        self.tree = tree

    # Returns the classification of a specific set of features
    def navigate_tree(self, test_input):
        current_node = self.tree
        while current_node.label == False:
            attribute = current_node.attribute
            value = test_input[attribute]
            current_node = current_node.children[value]
        classification = current_node.classification
        return classification


# F1 score function
def f1_score_calc(tp, tn, fp, fn):
    precision = float(tp) / (float(tp) + float(fp))
    recall = float(tp) / (float(tp) + float(fn))
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


# Determines the f1 scores for each validation range
def cross_validate(train, attributes, importance):
    validation_ranges = [(0, 54), (55, 109), (110, 164), (165, 219), (220, 274),
                         (275, 329), (330, 384), (385, 439), (440, 494), (495, 549)]
    f1_scores = {}
    for validation_range in validation_ranges:
        new_train = []
        validation_set = []
        for i in range(550):
            if i >= validation_range[0] and i <= validation_range[1]:
                validation_set.append(train[i])
            else:
                new_train.append(train[i])
        tree = DecisionTree(importance)
        temp_tree = tree.create_tree(new_train, attributes, new_train)
        tree.add_tree(temp_tree)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for validation_example in validation_set:
            prediction = tree.navigate_tree(validation_example.features)
            if validation_example.label == 'positive':
                if prediction == 'positive':
                    tp = tp + 1
                elif prediction == 'negative':
                    fn = fn + 1
            elif validation_example.label == 'negative':
                if prediction == 'positive':
                    fp = fp + 1
                elif prediction == 'negative':
                    tn = tn + 1
        f1_scores[validation_range] = f1_score_calc(tp, tn, fp, fn)
    return f1_scores

# Performs overvall validation for the trees, finds the best and uses test data
def evaluate_trees(gain_train, gain_test, gain_attributes, gini_train, gini_test, gini_attributes):
    f1_scores_gain_ratio = cross_validate(gain_train, gain_attributes, 0)
    f1_scores_gini_index = cross_validate(gini_train, gini_attributes, 1)
    validation_ranges = f1_scores_gain_ratio.keys()
    max_f1_gain_ratio, max_f1_gini_index = 0, 0
    max_gain_ratio_validation, max_gini_index_validation = [], []
    print("Gain Ratio F1 Scores")
    for key in validation_ranges:
        if f1_scores_gain_ratio[key] > max_f1_gain_ratio:
            max_f1_gain_ratio = f1_scores_gain_ratio[key]
            max_gain_ratio_validation = key
        print("Validation Range: ", key, "\t", "F1_Score: ", f1_scores_gain_ratio[key])
    print("Gini Index F1 Scores")
    for key in validation_ranges:
        if f1_scores_gini_index[key] > max_f1_gini_index:
            max_f1_gini_index = f1_scores_gini_index[key]
            max_gini_index_validation = key
        print("Validation Range: ", key, "\t", "F1_Score: ", f1_scores_gini_index[key])
    print("\n")
    print("Best validation range for gain ratio: ", max_gain_ratio_validation)
    print("Best validation range for gini index: ", max_gini_index_validation)
    gain_ratio_tree = DecisionTree(0)
    gini_index_tree = DecisionTree(1)
    gain_ratio_train = []
    gini_index_train = []
    for i in range(550):
        if i >= max_gain_ratio_validation[0] and i <= max_gain_ratio_validation[1]:
            continue
        else:
            gain_ratio_train.append(gain_train[i])
    for i in range(550):
        if i >= max_gini_index_validation[0] and i <= max_gini_index_validation[1]:
            continue
        else:
            gini_index_train.append(gini_train[i])
    gain_tp, gain_tn, gain_fp, gain_fn = 0, 0, 0, 0
    gini_tp, gini_tn, gini_fp, gini_fn = 0, 0, 0, 0
    gain_ratio_tree.add_tree(gain_ratio_tree.create_tree(gain_ratio_train, gain_attributes, gain_ratio_train))
    gini_index_tree.add_tree(gini_index_tree.create_tree(gini_index_train, gini_attributes, gini_index_train))
    for test_example in gain_test:
        gain_prediction = gain_ratio_tree.navigate_tree(test_example.features)
        if test_example.label == 'positive':
            if gain_prediction == 'positive':
                gain_tp = gain_tp + 1
            elif gain_prediction == 'negative':
                gain_fn = gain_fn + 1
        if test_example.label == 'negative':
            if gain_prediction == 'positive':
                gain_fp = gain_fp + 1
            elif gain_prediction == 'negative':
                gain_tn = gain_tn + 1
    for test_example in gini_test:
        gini_prediction = gini_index_tree.navigate_tree(test_example.features)
        if test_example.label == 'positive':
            if gini_prediction == 'positive':
                gini_tp = gini_tp + 1
            elif gini_prediction == 'negative':
                gini_fn = gini_fn + 1
        if test_example.label == 'negative':
            if gini_prediction == 'positive':
                gini_fp = gini_fp + 1
            elif gini_prediction == 'negative':
                gini_tn = gini_tn + 1
    gain_ratio_f1_score = f1_score_calc(gain_tp, gain_tn, gain_fp, gain_fn)
    gini_index_f1_score = f1_score_calc(gini_tp, gini_tn, gini_fp, gini_fn)
    print("\n")
    print("F1 score for gain ratio on test: ", gain_ratio_f1_score)
    print("F1 score for gini index on test: ", gini_index_f1_score)


# I found the following function as a solution on stack overflow: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
# It's so simple I figured it wasn't really an issue to copy
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# pull test data from file
def pull_test_data():
    lines = []
    with open("test.data") as test:
        for line in test:
            lines.append(line.rstrip())
    return lines

# pull train data from file
def pull_train_data():
    lines = []
    with open("training.data") as train:
        for line in train:
            lines.append(line.rstrip())
    return lines

# splits test data into attributes by line
def process_test_data(lines):
    test_lines = []
    for line in lines:
        test_lines.append(line.split(","))
    return test_lines

# Split the training data and determine attribute types, medians, and perform replacement of missing values
def process_training_data(lines):
    attributes = []
    attribute_types = []
    new_lines = []
    for line in lines:
        new_lines.append(line.split(","))
    for i in range(len(new_lines[0]) - 1):
        attributes.append(i)
        if is_number(new_lines[0][i]):
            attribute_types.append('c')
        else:
            attribute_types.append('d')
    values_present = []
    for i in range(len(attributes)):
        values_present.append([])
    for line in new_lines:
        for i in range(len(line) - 1):
            if line[i] != '?':
                values_present[i].append(line[i])
    medians = []
    for values_list in values_present:
        values_list.sort()
        index = len(values_list) // 2
        medians.append(values_list[index])
    for line in new_lines:
        for i in range(len(attributes) - 1):
            if line[i] == '?':
                line[i] = medians[i]
            if attribute_types[i] == 'c':
                line[i] = float(line[i])
    return medians, new_lines, attribute_types, attributes

# Find all possible split points
def determine_split_points(lines, i):
    values = []
    split_points = []
    for line in lines:
        values.append(line[i])
    for j in range(len(values) - 1):
        sum = float(values[j]) + float(values[j + 1])
        avg = sum / 2
        split_points.append(avg)
    return split_points

# convert input into format that the ree function can manage, remove continuous values and rename labels, and convert to example objects
# also convert to attribute objects
def final_input_manage(lines, attribute_types, attributes, split_map):
    values_map = {}
    final_attributes = []
    final_examples = []
    raw_lines = copy.deepcopy(lines)
    final_attributes = []
    for i in range(len(attribute_types)):
        if attribute_types[i] == 'c':
            values_map[i] = ['high', 'low']
        else:
            values_map[i] = []
    for line in raw_lines:
        feature_map = {}
        label = ''
        for i in range(len(line)):
            if line[i] == '+':
                line[i] = 'positive'
                label = 'positive'
            elif line[i] == '-':
                line[i] = 'negative'
                label = 'negative'
            elif attribute_types[i] == 'c':
                if float(line[i]) < split_map[i]:
                    feature_map[i] = 'low'
                    line[i] = 'low'
                else:
                    feature_map[i] = 'high'
                    line[i] = 'high'
            elif attribute_types[i] == 'd':
                feature_map[i] = line[i]
                if line[i] not in values_map[i]:
                    values_map[i].append(line[i])
        temp_example = Example(feature_map, label)
        final_examples.append(temp_example)
    for key in values_map.keys():
        temp_attribute = Attribute(key, values_map[key])
        final_attributes.append(temp_attribute)
    return final_attributes, final_examples

# find the entropy of a set of lines
def find_entropy(lines):
    positives = 0
    negatives = 0
    for line in lines:
        if line[-1] == '+':
            positives = positives + 1
        elif line[-1] == '-':
            negatives = negatives + 1
    pos_ratio = float(positives) / float(positives + negatives)
    neg_ratio = float(negatives) / float(positives + negatives)
    if pos_ratio == 0:
        temp_pos = 0
    else:
        temp_pos = pos_ratio * math.log2(pos_ratio)
    if neg_ratio == 0:
        temp_neg = 0
    else:
        temp_neg = neg_ratio * math.log2(neg_ratio)
    sum = temp_pos + temp_neg
    temp = sum * -1
    return temp

# find the gini value of a set of lines
def find_gini_value(lines):
    positives = 0
    negatives = 0
    for line in lines:
        if line[-1] == '+':
            positives = positives + 1
        elif line[-1] == '-':
            negatives = negatives + 1
    pos_ratio = float(positives) / (positives + negatives)
    neg_ratio = float(negatives) / (positives + negatives)
    pos_squared = pos_ratio ** 2
    neg_squared = neg_ratio ** 2
    sum = pos_squared + neg_squared
    gini_value = 1 - sum
    return gini_value

# find the gain ratio at a certain split point
def find_gain_ratio(split_point, attribute, lines):
    low = []
    high = []
    for line in lines:
        if float(line[attribute]) > split_point:
            high.append(line)
        else:
            low.append(line)
    entropy_dataset = find_entropy(lines)
    entropy_high = find_entropy(high)
    entropy_low = find_entropy(low)
    high_count = float(len(high))
    low_count = float(len(low))
    total_count = float(len(lines))
    temp0 = entropy_high * abs(high_count / total_count)
    temp1 = entropy_low * abs(low_count / total_count)
    sum = temp0 + temp1
    information_gain = entropy_dataset - sum
    intrinsic_value = -1 * (((high_count / total_count) * math.log2(high_count / total_count)) + (
                (low_count / total_count) * math.log2(low_count / total_count)))
    gain_ratio = information_gain / intrinsic_value
    return gain_ratio

# find the gini index at a certain split point
def find_gini_index(split_point, attribute, lines):
    low = []
    high = []
    for line in lines:
        if float(line[attribute]) > split_point:
            high.append(line)
        else:
            low.append(line)
    gini_low = find_gini_value(low)
    gini_high = find_gini_value(high)
    low_ratio = abs(float(len(low)) / float(len(lines)))
    high_ratio = abs(float(len(high)) / float(len(lines)))
    temp0 = low_ratio * gini_low
    temp1 = high_ratio * gini_high
    gini_index = temp0 + temp1
    return gini_index

# determine the best split points for gain ratio
def determine_gain_split_points(split_map, lines):
    gain_ratio_max = 0
    final_map = {}
    for key in split_map.keys():
        for split_point in split_map[key]:
            gain_ratio_current = find_gain_ratio(split_point, key, lines)
            if gain_ratio_current > gain_ratio_max:
                gain_ratio_max = gain_ratio_current
                final = split_point
        final_map[key] = final
    return final_map

# determine the best split points for gini index
def determine_gini_split_points(split_map, lines):
    gini_index_min = 100
    final_map = {}
    for key in split_map.keys():
        for split_point in split_map[key]:
            gini_index_current = find_gini_index(split_point, key, lines)
            if gini_index_current < gini_index_min:
                gini_index_min = gini_index_current
                final = split_point
        final_map[key] = split_point
    return final_map

# remove blanks from test data and convert to example objects
def manage_test_data(lines, split_map, attribute_types, medians):
    test_examples = []
    new_lines = copy.deepcopy(lines)
    for line in new_lines:
        temp_features = {}
        label = ''
        for i in range(len(line)):
            if line[i] == '?':
                line[i] = medians[i]
            if line[i] == '-':
                label = 'negative'
            elif line[i] == '+':
                label = 'positive'
            elif attribute_types[i] == 'd':
                temp_features[i] = line[i]
            elif attribute_types[i] == 'c':
                if float(line[i]) < split_map[i]:
                    temp_features[i] = 'low'
                else:
                    temp_features[i] = 'high'
        temp_example = Example(temp_features, label)
        test_examples.append(temp_example)
    return test_examples


# overall manage input function for reading in and processing input
def manage_input():
    train_lines = pull_train_data()
    test_lines = pull_test_data()
    medians, processed_training, attribute_types, attributes = process_training_data(train_lines)
    processed_test_lines = process_test_data(test_lines)
    split_point_map = {}
    for i in range(len(attribute_types)):
        if attribute_types[i] == 'c':
            split_point_map[i] = determine_split_points(processed_training, i)
    gain_split_points = determine_gain_split_points(split_point_map, processed_training)
    gini_split_points = determine_gini_split_points(split_point_map, processed_training)
    gain_attributes, gain_examples = final_input_manage(processed_training, attribute_types, attributes,
                                                        gain_split_points)
    gini_attributes, gini_examples = final_input_manage(processed_training, attribute_types, attributes,
                                                        gini_split_points)
    gain_test = manage_test_data(processed_test_lines, gain_split_points, attribute_types, medians)
    gini_test = manage_test_data(processed_test_lines, gini_split_points, attribute_types, medians)
    return gain_examples, gain_test, gain_attributes, gini_examples, gini_test, gini_attributes


def main():
    gain_train, gain_test, gain_attributes, gini_train, gini_test, gini_attributes = manage_input()
    evaluate_trees(gain_train, gain_test, gain_attributes, gini_train, gini_test, gini_attributes)


if __name__ == '__main__':
    main()