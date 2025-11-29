# Example AWS IAM Snippets

Production references for setting up S3 SSE-KMS encryption, KMS key usage, and a placeholder for DynamoDB-style access if you wire a managed StatusStore.

Note: Adapt ARNs and principals to your environment (account IDs, roles). Use least privilege.

## 1) S3 bucket policy requiring SSE-KMS

Enforce that all writes are encrypted with your KMS key and deny unencrypted PUTs or use of other keys.

```json
{
  "Version": "2012-10-17",
  "Id": "S3BucketPolicyRequireKMS",
  "Statement": [
    {
      "Sid": "DenyUnEncryptedObjectUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::your-artifacts-bucket/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    },
    {
      "Sid": "DenyWrongKMSKey",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::your-artifacts-bucket/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption-aws-kms-key-id": "arn:aws:kms:us-east-1:123456789012:key/your-kms-key-id"
        }
      }
    },
    {
      "Sid": "AllowAppRolesReadWrite",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::123456789012:role/prod-gateway",
          "arn:aws:iam::123456789012:role/prod-explainer"
        ]
      },
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:AbortMultipartUpload",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-artifacts-bucket",
        "arn:aws:s3:::your-artifacts-bucket/*"
      ]
    }
  ]
}
```

## 2) KMS key policy for the service principals

Permit the service roles to encrypt/decrypt with the key used for SSE-KMS. Prefer using a key alias in app config; keep rotation enabled.

```json
{
  "Version": "2012-10-17",
  "Id": "KeyPolicyForHypergraph",
  "Statement": [
    {
      "Sid": "EnableIAMUserPermissions",
      "Effect": "Allow",
      "Principal": { "AWS": "arn:aws:iam::123456789012:root" },
      "Action": "kms:*",
      "Resource": "*"
    },
    {
      "Sid": "AllowAppRolesUseOfKey",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::123456789012:role/prod-gateway",
          "arn:aws:iam::123456789012:role/prod-explainer"
        ]
      },
      "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:ReEncrypt*",
        "kms:GenerateDataKey*",
        "kms:DescribeKey"
      ],
      "Resource": "*"
    }
  ]
}
```

## 3) Application role policy for S3

Attach to the workload role (e.g., prod-explainer) to allow read/write to the bucket with KMS-encrypted objects.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3RWArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:AbortMultipartUpload",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-artifacts-bucket",
        "arn:aws:s3:::your-artifacts-bucket/*"
      ]
    },
    {
      "Sid": "UseKMSForSSE",
      "Effect": "Allow",
      "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:ReEncrypt*",
        "kms:GenerateDataKey*",
        "kms:DescribeKey"
      ],
      "Resource": "arn:aws:kms:us-east-1:123456789012:key/your-kms-key-id"
    }
  ]
}
```

## 4) Placeholder DynamoDB access (Status/Index Store)

If you later migrate the LocalJSON status store to DynamoDB, use a tightly scoped policy (table name, partition key only). This is a placeholder—adapt to your schema.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DDBAccessStatusStore",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "dynamodb:DescribeTable"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/hypergraph-status"
    }
  ]
}
```

## Operator notes

- Prefer key aliases (e.g., alias/hypergraph-artifacts) and enable automatic KMS rotation.
- Deny public access on the S3 bucket; enable Block Public Access.
- Attach separate roles for Gateway vs Explainer if blast radius reduction is required (principle of least privilege).
- Ensure VPC endpoints for S3/DynamoDB are used to avoid public egress where possible.