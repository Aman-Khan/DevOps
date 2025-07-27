pipeline {
  agent {
    label 'slave-node'
  }

  environment {
    GIT_CREDENTIALS_ID = 'JenkinsTest'
  }

  stages {
    stage('Info') {
      steps {
        echo "Building branch: ${env.BRANCH_NAME}"
      }
    }

    stage('Deploy') {
      steps {
        script {
          if (env.BRANCH_NAME == 'main') {
            echo "Deploying to production server..."
            sh '''
            cp WebPage/index.html /var/www/html/index.html
            sudo systemctl restart apache2
            '''
          } else {
            echo "Not production — only verifying HTML exists."
            sh 'ls -l WebPage/index.html'
          }
        }
      }
    }
  }

  post {
    always {
      echo "Pipeline done for branch: ${env.BRANCH_NAME}"
    }
  }
}
