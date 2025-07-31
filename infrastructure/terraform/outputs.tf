# Terraform outputs for Agent Mesh infrastructure

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_token" {
  description = "EKS cluster authentication token"
  value       = data.aws_eks_cluster_auth.cluster.token
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs"
  value       = module.vpc.natgw_ids
}

output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = aws_route53_zone.main.zone_id
}

output "route53_zone_name_servers" {
  description = "Route53 zone name servers"
  value       = aws_route53_zone.main.name_servers
}

output "load_balancer_controller_role_arn" {
  description = "AWS Load Balancer Controller IAM role ARN"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

output "external_dns_role_arn" {
  description = "External DNS IAM role ARN"
  value       = module.external_dns_irsa_role.iam_role_arn
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : null
  sensitive   = true
}

output "database_port" {
  description = "RDS database port"
  value       = var.enable_rds ? aws_db_instance.main[0].port : null
}

output "kubectl_config" {
  description = "kubectl configuration command"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${module.eks.cluster_name}"
}

output "grafana_url" {
  description = "Grafana dashboard URL (after port-forward)"
  value       = "http://localhost:3000 (kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80)"
}

output "prometheus_url" {
  description = "Prometheus URL (after port-forward)"
  value       = "http://localhost:9090 (kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090)"
}