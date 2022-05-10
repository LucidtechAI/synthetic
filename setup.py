import pathlib
from setuptools import setup


with open('requirements.txt') as fp:
    install_requires = fp.readlines()


with open('README.md') as fp:
    readme = fp.read()


about = {}
version_file_path = pathlib.Path(__file__).parent / 'synthetic' / '__version__.py'
exec(version_file_path.read_text('utf-8'), about)


setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    maintainer=about['__maintainer__'],
    maintainer_email=about['__maintainer_email__'],
    url=about['__url__'],
    packages=[
        'synthetic',
        'synthetic.core',
        'synthetic.pdf',
    ],
    entry_points={
        'console_scripts': [
            'synthetic = synthetic.__main__:main'
        ]
    },
    install_requires=install_requires,
    platforms='Posix; MacOS X; Windows',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Internet'
    ]
)
