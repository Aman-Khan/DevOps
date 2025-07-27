pipeline {
  agent any

  stages {
    stage('Info') {
      steps {
        echo "Running branch: ${env.BRANCH_NAME}"
      }
    }

    stage('Build') {
      steps {
        script {
          if (env.BRANCH_NAME == 'main') {
            echo "✅ This is main — deploying to Production!"
            // your prod deploy steps
          } else if (env.BRANCH_NAME == 'develop') {
            echo "🧪 This is develop — deploying to Staging!"
            // your staging deploy steps
          } else {
            echo "✨ This is feature branch — running tests only."
            // your test steps
          }
        }
      }
    }
  }

  post {
    always {
      echo "Pipeline for ${env.BRANCH_NAME} finished!"
    }
  }
}
