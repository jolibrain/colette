pipeline {
  agent none

  options {
    timestamps()
    disableConcurrentBuilds(abortPrevious: true)
    buildDiscarder(logRotator(numToKeepStr: '30'))
  }

  parameters {
    string(name: 'GPU_ID', defaultValue: '1', description: 'GPU index to expose in CUDA_VISIBLE_DEVICES')
    booleanParam(name: 'RUN_INTEGRATION_STABLE', defaultValue: false, description: 'Run optional protected integration-stable lane')
    booleanParam(name: 'RUN_NIGHTLY_MATRIX', defaultValue: false, description: 'Run full GPU matrix (integration, pipeline, e2e)')
  }

  environment {
    PYTHON_BIN = 'venv_colette/bin/python'
    CI_ARTIFACTS_DIR = '.ci-artifacts'
  }

  stages {
    stage('PR Smoke') {
      agent {
        node { label 'linux && cpu' }
      }
      steps {
        sh 'set -e; make ci-smoke'
      }
      post {
        always {
          archiveArtifacts artifacts: '.ci-artifacts/junit-smoke.xml', allowEmptyArchive: true
        }
      }
    }

    stage('Integration Stable (Optional)') {
      when {
        expression { return params.RUN_INTEGRATION_STABLE }
      }
      agent {
        node { label 'linux && gpu' }
      }
      steps {
        withCredentials([string(credentialsId: 'hf', variable: 'HF_TOKEN')]) {
          lock(resource: null, label: "${NODE_NAME}-gpu", variable: 'LOCKED_GPU', quantity: 1) {
            sh '''
              set -e
              export CUDA_VISIBLE_DEVICES=$(echo ${LOCKED_GPU} | sed -n -e "s/[^,]* GPU \\([^[0-9,]]\\)*/\\1/gp")
              [ -n "$CUDA_VISIBLE_DEVICES" ] || export CUDA_VISIBLE_DEVICES=${GPU_ID}
              export COLETTE_GPU_ID=$CUDA_VISIBLE_DEVICES
              export COLETTE_RUN_INTEGRATION=1
              export HF_TOKEN=${HF_TOKEN}
              nvidia-smi
              make ci-integration-stable GPU_ID=$CUDA_VISIBLE_DEVICES
            '''
          }
        }
      }
      post {
        always {
          archiveArtifacts artifacts: '.ci-artifacts/junit-integration-stable.xml', allowEmptyArchive: true
        }
      }
    }

    stage('Nightly GPU Matrix') {
      when {
        expression { return params.RUN_NIGHTLY_MATRIX }
      }
      parallel {
        stage('Integration') {
          agent {
            node { label 'linux && gpu' }
          }
          steps {
            withCredentials([string(credentialsId: 'hf', variable: 'HF_TOKEN')]) {
              lock(resource: null, label: "${NODE_NAME}-gpu", variable: 'LOCKED_GPU', quantity: 1) {
                sh '''
                  set -e
                  export CUDA_VISIBLE_DEVICES=$(echo ${LOCKED_GPU} | sed -n -e "s/[^,]* GPU \\([^[0-9,]]\\)*/\\1/gp")
                  [ -n "$CUDA_VISIBLE_DEVICES" ] || export CUDA_VISIBLE_DEVICES=${GPU_ID}
                  export COLETTE_GPU_ID=$CUDA_VISIBLE_DEVICES
                  export COLETTE_RUN_INTEGRATION=1
                  export HF_TOKEN=${HF_TOKEN}
                  nvidia-smi
                  make ci-integration GPU_ID=$CUDA_VISIBLE_DEVICES
                '''
              }
            }
          }
          post {
            always {
              archiveArtifacts artifacts: '.ci-artifacts/junit-integration.xml', allowEmptyArchive: true
            }
          }
        }

        stage('Pipeline Integration') {
          agent {
            node { label 'linux && gpu' }
          }
          steps {
            withCredentials([string(credentialsId: 'hf', variable: 'HF_TOKEN')]) {
              lock(resource: null, label: "${NODE_NAME}-gpu", variable: 'LOCKED_GPU', quantity: 1) {
                sh '''
                  set -e
                  export CUDA_VISIBLE_DEVICES=$(echo ${LOCKED_GPU} | sed -n -e "s/[^,]* GPU \\([^[0-9,]]\\)*/\\1/gp")
                  [ -n "$CUDA_VISIBLE_DEVICES" ] || export CUDA_VISIBLE_DEVICES=${GPU_ID}
                  export COLETTE_GPU_ID=$CUDA_VISIBLE_DEVICES
                  export COLETTE_RUN_INTEGRATION=1
                  export HF_TOKEN=${HF_TOKEN}
                  nvidia-smi
                  make ci-pipeline-integration GPU_ID=$CUDA_VISIBLE_DEVICES
                '''
              }
            }
          }
          post {
            always {
              archiveArtifacts artifacts: '.ci-artifacts/junit-pipeline-integration.xml', allowEmptyArchive: true
            }
          }
        }

        stage('E2E') {
          agent {
            node { label 'linux && gpu' }
          }
          steps {
            withCredentials([string(credentialsId: 'hf', variable: 'HF_TOKEN')]) {
              lock(resource: null, label: "${NODE_NAME}-gpu", variable: 'LOCKED_GPU', quantity: 1) {
                sh '''
                  set -e
                  export CUDA_VISIBLE_DEVICES=$(echo ${LOCKED_GPU} | sed -n -e "s/[^,]* GPU \\([^[0-9,]]\\)*/\\1/gp")
                  [ -n "$CUDA_VISIBLE_DEVICES" ] || export CUDA_VISIBLE_DEVICES=${GPU_ID}
                  export COLETTE_GPU_ID=$CUDA_VISIBLE_DEVICES
                  export COLETTE_RUN_INTEGRATION=1
                  export HF_TOKEN=${HF_TOKEN}
                  nvidia-smi
                  make ci-e2e GPU_ID=$CUDA_VISIBLE_DEVICES
                '''
              }
            }
          }
          post {
            always {
              archiveArtifacts artifacts: '.ci-artifacts/junit-e2e.xml', allowEmptyArchive: true
            }
          }
        }
      }
    }
  }

  post {
    always {
      cleanWs(
        cleanWhenAborted: true,
        cleanWhenFailure: true,
        cleanWhenNotBuilt: true,
        cleanWhenSuccess: true,
        cleanWhenUnstable: true,
        cleanupMatrixParent: true,
        deleteDirs: true,
      )
    }
  }
}
