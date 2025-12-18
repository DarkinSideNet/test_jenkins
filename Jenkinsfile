pipeline {
    agent any

    environment {
        // --- CẤU HÌNH AWS ---
        AWS_REGION = 'us-east-1'
        // AMI Ubuntu 22.04 LTS (Deep Learning Base AMI thì bạn thay ID khác)
        EC2_AMI_ID = 'ami-0c398cb65a93047f2' 
        EC2_INSTANCE_TYPE = 't3.small'
        
        // Key Pair name ĐÃ TẠO TRÊN AWS CONSOLE
        EC2_KEY_NAME = 'eks-key' 
        // ID của Security Group (phải mở port 22)
        EC2_SG_ID = 'sg-0677b9b15b8711d14' 

        // ID Credential lưu trong Jenkins (chứa file PEM)
        JENKINS_SSH_CRED_ID = 'ec2-key-pem' 
    }

    stages {
        stage('1. Launch EC2 Instance') {
            steps {
                // BƯỚC QUAN TRỌNG: Load AWS Key vào biến môi trường
                withCredentials([usernamePassword(credentialsId: AWS_CRED_ID, passwordVariable: 'AWS_SECRET_ACCESS_KEY', usernameVariable: 'AWS_ACCESS_KEY_ID')]) {
                    script {
                        echo "Launching EC2 Instance..."
                        
                        // Lúc này biến môi trường AWS_ACCESS_KEY_ID đã có giá trị
                        // Lệnh aws cli sẽ tự động nhận diện nó.
                        def output = sh(returnStdout: true, script: """
                            aws ec2 run-instances \
                                --image-id ${EC2_AMI_ID} \
                                --count 1 \
                                --instance-type ${EC2_INSTANCE_TYPE} \
                                --key-name ${EC2_KEY_NAME} \
                                --security-group-ids ${EC2_SG_ID} \
                                --region ${AWS_REGION} \
                                --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Jenkins-Training-Worker}]' \
                                --query 'Instances[0].InstanceId' \
                                --output text
                        """).trim()
                        
                        env.INSTANCE_ID = output
                        echo "Instance Created: ${env.INSTANCE_ID}"
                    }
                }
            }
        }

        // stage('2. Wait for IP & SSH Ready') {
        //     steps {
        //         script {
        //             echo "⏳ Waiting for Instance to be RUNNING..."
        //             sh "aws ec2 wait instance-running --instance-ids ${env.INSTANCE_ID} --region ${AWS_REGION}"

        //             // Lấy Public IP
        //             env.INSTANCE_IP = sh(returnStdout: true, script: """
        //                 aws ec2 describe-instances \
        //                     --instance-ids ${env.INSTANCE_ID} \
        //                     --region ${AWS_REGION} \
        //                     --query 'Reservations[0].Instances[0].PublicIpAddress' \
        //                     --output text
        //             """).trim()
                    
        //             echo "🌐 Public IP: ${env.INSTANCE_IP}"
                    
        //             // Chờ thêm 60s để SSH Daemon trên máy Ubuntu kịp khởi động
        //             echo "💤 Sleeping 60s for SSH Daemon to start..."
        //             sleep 60
        //         }
        //     }
        // }

        // stage('3. SSH & Execute Training') {
        //     steps {
        //         // Load file PEM từ Jenkins Credential vào biến file
        //         sshagent(credentials: [JENKINS_SSH_CRED_ID]) {
        //             script {
        //                 echo "🔌 Connecting via SSH..."
                        
        //                 // Cấu hình SSH: 
        //                 // -o StrictHostKeyChecking=no: Để không hỏi Yes/No khi connect lần đầu
        //                 // ubuntu@${INSTANCE_IP}: User mặc định của AMI Ubuntu
                        
        //                 def remoteCommand = """
        //                     echo '--- HELLO FROM EC2 G4DN ---'
        //                     hostname
        //                     whoami
                            
        //                     echo '--- CHECKING GPU ---'
        //                     # Kiểm tra xem có lệnh nvidia-smi không (nếu dùng AMI thường sẽ chưa có)
        //                     if command -v nvidia-smi &> /dev/null; then
        //                         nvidia-smi
        //                     else
        //                         echo 'WARNING: Nvidia Driver not found. Please use Deep Learning AMI.'
        //                     fi

        //                     echo '--- SIMULATING TRAINING ---'
        //                     mkdir -p workspace
        //                     cd workspace
        //                     echo 'Cloning git...'
        //                     # git clone ... (Điền lệnh git của bạn vào đây)
                            
        //                     echo 'Training...'
        //                     # python3 train.py ...
        //                     sleep 10 # Giả lập đang train
                            
        //                     echo '--- DONE ---'
        //                 """

        //                 // Thực thi lệnh từ xa
        //                 sh "ssh -o StrictHostKeyChecking=no ubuntu@${env.INSTANCE_IP} \"${remoteCommand}\""
        //             }
        //         }
        //     }
        // }
    }

    // Khối này LUÔN LUÔN chạy dù các bước trên có lỗi hay không
    // post {
    //     always {
    //         script {
    //             echo "🛑 TERMINATING INSTANCE ${env.INSTANCE_ID}..."
    //             // Kiểm tra nếu biến INSTANCE_ID có giá trị thì mới xóa
    //             if (env.INSTANCE_ID) {
    //                 sh "aws ec2 terminate-instances --instance-ids ${env.INSTANCE_ID} --region ${AWS_REGION}"
    //                 echo "✅ Instance terminated."
    //             }
    //         }
    //     }
    //     failure {
    //         echo "❌ Pipeline Failed! Check logs."
    //     }
    // }
}