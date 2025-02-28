from setuptools import setup, find_packages

setup(
    name='samecode',
    version='1.0.0',
    author='Gustavo Arango',
    author_email='',
    description='everyday tools',
    packages=find_packages(),
    install_requires=[
        # List your module's dependencies here
    ],
    entry_points={
        'console_scripts': [
            'generate_project=samecode.project_manager.create_project:main',
        ],
    },
)
