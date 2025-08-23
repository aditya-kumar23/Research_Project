from setuptools import setup

package_name = 'orchestrator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # ament index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        # package manifest
        (
            'share/' + package_name,
            ['package.xml']
        ),
        # all launch scripts
        (
            'share/' + package_name + '/launch',
            [
                'launch/orchestrator.launch.py',
                'launch/multi_agent_centralized.launch.py'
            ]
        ),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Multi-agent orchestrator for collaborative SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # (none for this package)
        ],
    },
)

