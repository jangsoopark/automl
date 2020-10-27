import jinja2
import os


class PascalVocWriter(object):

    def __init__(self, path, width, height, depth=3, database='UnKnown', segmented=0):
        environment = jinja2.Environment(
            loader=jinja2.PackageLoader('pascal', 'templates'),
            keep_trailing_newline=True
        )
        self.template = environment.get_template('annotation.xml')
        abspath = os.path.abspath(path)
        self.context = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def add_object(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.context['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, path):
        with open(path, 'w') as f:
            content = self.template.render(**self.context)
            f.write(content)
