"""
This module defines export functions for decision trees.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Trevor Stephens <trev.stephens@gmail.com>
#          Li Li <aiki.nogard@gmail.com>
#          Giuseppe Vettigli <vettigli@gmail.com>
#          Gustavo Arango <gaarangoa@gmail.com>
# License: BSD 3 clause
from collections.abc import Iterable
from io import StringIO
from numbers import Integral
from sklearn.base import is_classifier
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.tree._reingold_tilford import Tree, buchheim
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from sklearn.tree._export import Sentinel, _BaseTreeExporter

import numpy as np 

SENTINEL = Sentinel()

@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rounded": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "ax": "no_validation",  # delegate validation to matplotlib
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)

def plot_tree2(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    class_label_colors=None,
    arrow_y_offset=0,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    class_colors = None,
):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of str or True, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)
    [...]
    """

    check_is_fitted(decision_tree)

    exporter = _MPLTreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        class_colors = class_colors,
        class_label_colors = class_label_colors,
        arrow_y_offset=arrow_y_offset,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
    )
    return exporter.export(decision_tree, ax=ax)


class _MPLTreeExporter(_BaseTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        class_colors=None,
        class_label_colors=None,
        arrow_y_offset = 0,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.fontsize = fontsize
        self.arrow_y_offset = arrow_y_offset
        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        self.max_n = -np.inf
        self.nodes_info = {}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="-")

        try:
            self.class_colors = {i:j for i,j in zip(class_names, class_colors)}
        except:
            self.class_colors = {}

        try:
            self.class_label_colors = {i:j for i,j in zip(class_names, class_label_colors)}
        except:
            self.class_label_colors = {}

        self.end_nodes = {}
        
    def node_to_str(self, tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "x%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )

        # print(node_string)

        # Write node sample count
        if labels:
            node_string += ""
        if self.proportion:
            percent = (
                100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]

        # Write node majority class
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                node_string += ""
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            node_string += class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        # print(node_string + characters[5])
        
        return node_string
    
    def _make_tree(self, node_id, et, criterion, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, node_id, criterion=criterion)

        label_ = name.split('\n')
        if len(label_) == 3: 
            node_name, counts, label = label_
            node_name = f'{node_name} ({counts})'
            self.end_nodes[node_id] = False
        else:
            counts, label = label_
            node_name = f'{label} ({counts})'
            self.end_nodes[node_id] = True

        if int(counts) > self.max_n:
            self.max_n = int(counts)
        
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            children = [
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1
                ),
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1
                ),
            ]
        else:
            Tree_ = Tree(node_name, node_id)
            Tree_.label__ = label
            self.nodes_info[node_id] = int(counts)
            return Tree_
            
        Tree_ = Tree(node_name, node_id, *children)
        Tree_.label__ = label
        self.nodes_info[node_id] = int(counts)
        return Tree_
        
    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]
        
        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [
                bbox_patch.get_window_extent()
                for ann in anns
                if (bbox_patch := ann.get_bbox_patch()) is not None
            ]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        import matplotlib.pyplot as plt

        # print(node.tree.label)
        # kwargs for annotations without a bounding box
        common_kwargs = dict(
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
        )
        if self.fontsize is not None:
            common_kwargs["fontsize"] = self.fontsize
        
        # kwargs for annotations with a bounding box
        kwargs = dict(
            ha="center",
            va="center",
            bbox=self.bbox_args.copy(),
            arrowprops=self.arrow_args.copy(),
            **common_kwargs,
        )
        # kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]
        

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:

            kwargs['bbox']['edgecolor'] = 'none'
            alpha = self.nodes_info[node.tree.node_id] / self.max_n
            kwargs['bbox']['alpha'] = alpha if alpha > 0.1 else 0.1
            # kwargs['zorder'] = 2

            kwargs["bbox"]["fc"] = self.class_colors.get(node.tree.label__, 'white')
            # kwargs["color"] = self.class_label_colors.get(node.tree.label__, 'black')
            kwargs["color"] = 'white' if alpha > 0.7 else 'black'
            
            if self.end_nodes[node.tree.node_id]:
                # kwargs['alpha'] = alpha if alpha > 0.7 else 0.7
                kwargs["bbox"]["fc"] = 'white'
                kwargs["color"] = self.class_colors.get(node.tree.label__, 'black')
                # kwargs['zorder'] = 5

            kwargs["arrowprops"]["edgecolor"] = self.class_colors.get(node.tree.label__, 'black')
            # kwargs["arrowprops"]["alpha"] = alpha #if alpha > 0.1 else 0.1
            kwargs["arrowprops"]["linewidth"] = 5*alpha if 5*alpha > 0.3 else 2*0.3
            # kwargs["alpha"] = alpha if alpha > 0.5 else 0.5
            
            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.6 - self.arrow_y_offset) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)

                # Draw True/False labels if parent is root node
                if node.parent.parent is None:
                    # Adjust the position for the text to be slightly above the arrow
                    text_pos = (
                        (xy_parent[0] + xy[0]) / 2,
                        (xy_parent[1] + xy[1]) / 2,
                    )
                    # Annotate the arrow with the edge label to indicate the child
                    # where the sample-split condition is satisfied
                    if node.parent.left() == node:
                        label_text, label_ha = ("True  ", "right")
                    else:
                        label_text, label_ha = ("  False", "left")
                        
                    ax.annotate(label_text, text_pos, ha=label_ha, **common_kwargs)
                    
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
